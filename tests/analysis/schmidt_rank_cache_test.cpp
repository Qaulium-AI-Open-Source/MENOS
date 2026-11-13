// Targeted tests for the Schmidt-rank cache (src/analysis/schmidt_rank_internal.h;
// docs/adr/0028 sections 10.4 and 12).
//
// The cache's one promise is that a hit returns exactly the integer a fresh
// operator_schmidt_rank would return. These tests check that on gate
// matrices and on random dense matrices across every split, and check the
// key: different splits of one matrix, different matrices, and +0.0 against
// -0.0 must be different entries, while separately built copies of the same
// matrix must share one.

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "analysis/schmidt_rank_internal.h"
#include "ir/gate_table.h"

namespace menos::analysis::detail {
namespace {

struct Split {
    std::vector<std::size_t> left;
    std::vector<std::size_t> right;
};

// Every split of k operand positions into two non-empty sides, each listed
// in ascending order, as compute_schmidt_rank_bound builds them.
std::vector<Split> all_splits(std::size_t k) {
    std::vector<Split> splits;
    for (std::size_t mask = 1; mask + 1 < (std::size_t{1} << k); ++mask) {
        Split s;
        for (std::size_t p = 0; p < k; ++p) {
            ((mask >> p) & 1U) != 0 ? s.left.push_back(p) : s.right.push_back(p);
        }
        splits.push_back(s);
    }
    return splits;
}

std::vector<Complex> random_matrix(std::size_t k, std::mt19937_64& rng) {
    std::normal_distribution<double> normal;
    const std::size_t dim = std::size_t{1} << k;
    std::vector<Complex> m(dim * dim);
    for (Complex& z : m) {
        z = {normal(rng), normal(rng)};
    }
    return m;
}

// A rank-1 (product) k-qubit matrix: random rank-1 blocks Kronecker-multiplied.
std::vector<Complex> product_matrix(std::size_t k, std::mt19937_64& rng) {
    std::vector<Complex> m{1.0};
    std::size_t dim = 1;
    for (std::size_t q = 0; q < k; ++q) {
        const std::vector<Complex> one = random_matrix(1, rng);
        std::vector<Complex> next(dim * 2 * dim * 2);
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                for (std::size_t i = 0; i < 2; ++i) {
                    for (std::size_t j = 0; j < 2; ++j) {
                        next[(r * 2 + i) * dim * 2 + (c * 2 + j)] = m[r * dim + c] * one[i * 2 + j];
                    }
                }
            }
        }
        m = std::move(next);
        dim *= 2;
    }
    return m;
}

std::vector<Complex> gate(Opcode op, std::vector<double> params = {}) {
    return ir::generate_matrix(op, params);
}

TEST(SchmidtRankCacheTest, AHitReturnsExactlyWhatRecomputationReturns) {
    std::mt19937_64 rng(20261104);
    std::vector<std::pair<std::size_t, std::vector<Complex>>> matrices{
        {2, gate(Opcode::CX)},
        {2, gate(Opcode::CZ)},
        {2, gate(Opcode::SWAP)},
        {2, gate(Opcode::CP, {0.3})},
        {2, gate(Opcode::RZZ, {1.1})},
        {3, gate(Opcode::CCX)},
    };
    for (std::size_t k = 2; k <= 4; ++k) {
        for (int i = 0; i < 5; ++i) {
            matrices.emplace_back(k, random_matrix(k, rng));
            matrices.emplace_back(k, product_matrix(k, rng));
        }
    }
    SchmidtRankCache cache;
    std::uint64_t lookups = 0;
    for (const auto& [k, u] : matrices) {
        for (const Split& s : all_splits(k)) {
            const std::uint64_t fresh = operator_schmidt_rank(u, k, s.left, s.right);
            EXPECT_EQ(cache.rank(u, k, s.left, s.right), fresh);  // miss
            EXPECT_EQ(cache.rank(u, k, s.left, s.right), fresh);  // hit
            lookups += 2;
        }
    }
    EXPECT_EQ(cache.stats().hits + cache.stats().misses, lookups);
    EXPECT_EQ(cache.stats().hits, lookups / 2);
}

TEST(SchmidtRankCacheTest, KnownRanksThroughTheCache) {
    SchmidtRankCache cache;
    const Split left0{{0}, {1}};
    EXPECT_EQ(cache.rank(gate(Opcode::CX), 2, left0.left, left0.right), 2U);
    EXPECT_EQ(cache.rank(gate(Opcode::SWAP), 2, left0.left, left0.right), 4U);
    std::mt19937_64 rng(3);
    EXPECT_EQ(cache.rank(product_matrix(2, rng), 2, left0.left, left0.right), 1U);
}

TEST(SchmidtRankCacheTest, DifferentSplitsOfOneMatrixAreDifferentEntries) {
    const std::vector<Complex> ccx = gate(Opcode::CCX);
    SchmidtRankCache cache;
    for (const Split& s : all_splits(3)) {
        static_cast<void>(cache.rank(ccx, 3, s.left, s.right));
    }
    EXPECT_EQ(cache.stats().misses, all_splits(3).size());
    EXPECT_EQ(cache.stats().hits, 0U);
}

TEST(SchmidtRankCacheTest, SeparatelyBuiltCopiesOfOneMatrixShareAnEntry) {
    SchmidtRankCache cache;
    const std::vector<std::size_t> left{0};
    const std::vector<std::size_t> right{1};
    static_cast<void>(cache.rank(gate(Opcode::CX), 2, left, right));
    static_cast<void>(cache.rank(gate(Opcode::CX), 2, left, right));
    EXPECT_EQ(cache.stats().misses, 1U);
    EXPECT_EQ(cache.stats().hits, 1U);
    // A different matrix of the same size is a different entry.
    static_cast<void>(cache.rank(gate(Opcode::CZ), 2, left, right));
    EXPECT_EQ(cache.stats().misses, 2U);
}

TEST(SchmidtRankCacheTest, SignedZerosAreDifferentKeysWithTheSameRank) {
    std::vector<Complex> plus = gate(Opcode::CZ);
    std::vector<Complex> minus = plus;
    minus[1] = {-0.0, -0.0};  // plus[1] is +0.0
    ASSERT_EQ(plus[1], minus[1]);
    const std::vector<std::size_t> left{0};
    const std::vector<std::size_t> right{1};
    SchmidtRankCache cache;
    const std::uint64_t a = cache.rank(plus, 2, left, right);
    const std::uint64_t b = cache.rank(minus, 2, left, right);
    EXPECT_EQ(cache.stats().misses, 2U);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a, operator_schmidt_rank(minus, 2, left, right));
}

TEST(SchmidtRankCacheTest, KeysOfDifferentOperandCountsDoNotCollide) {
    // Same leading bytes in every field but k: a 2-qubit and a 3-qubit
    // identity split after operand 0.
    std::vector<Complex> id2(16, 0.0);
    std::vector<Complex> id3(64, 0.0);
    for (std::size_t i = 0; i < 4; ++i) {
        id2[i * 4 + i] = 1.0;
    }
    for (std::size_t i = 0; i < 8; ++i) {
        id3[i * 8 + i] = 1.0;
    }
    SchmidtRankCache cache;
    EXPECT_EQ(cache.rank(id2, 2, {0}, {1}), 1U);
    EXPECT_EQ(cache.rank(id3, 3, {0}, {1, 2}), 1U);
    EXPECT_EQ(cache.stats().misses, 2U);
}

TEST(SchmidtRankCacheTest, KeyFieldsAreFixedWidthSoPositionsCannotAlias) {
    // With one-byte fields a split listing position 124 would have looked like
    // the old '|' separator; four-byte fields make every split distinct. Two
    // splits of one 3-qubit matrix that differ only in their last position:
    std::mt19937_64 rng(5);
    const std::vector<Complex> u = random_matrix(3, rng);
    SchmidtRankCache cache;
    static_cast<void>(cache.rank(u, 3, {0, 1}, {2}));
    static_cast<void>(cache.rank(u, 3, {0, 2}, {1}));
    static_cast<void>(cache.rank(u, 3, {0}, {1, 2}));
    EXPECT_EQ(cache.stats().misses, 3U);
    EXPECT_EQ(cache.stats().hits, 0U);
}

TEST(SchmidtRankCacheTest, ACapThatIsReachedClearsTheCacheAndResultsStayExact) {
    // A 2-qubit key is 4 + 4 + 4 + 256 = 268 bytes; a cap of 600 holds two.
    SchmidtRankCache cache(600);
    const std::vector<std::size_t> left{0};
    const std::vector<std::size_t> right{1};
    const std::vector<Complex> cx = gate(Opcode::CX);
    const std::vector<Complex> cz = gate(Opcode::CZ);
    const std::vector<Complex> swap = gate(Opcode::SWAP);
    EXPECT_EQ(cache.rank(cx, 2, left, right), 2U);
    EXPECT_EQ(cache.rank(cz, 2, left, right), 2U);
    EXPECT_EQ(cache.stats().bytes, 536U);
    EXPECT_EQ(cache.rank(swap, 2, left, right), 4U);  // evicts
    EXPECT_EQ(cache.stats().evictions, 1U);
    EXPECT_EQ(cache.stats().bytes, 268U);
    EXPECT_EQ(cache.rank(swap, 2, left, right), 4U);  // hit
    EXPECT_EQ(cache.rank(cx, 2, left, right), 2U);    // recomputed after eviction
    EXPECT_EQ(cache.stats().hits, 1U);
    EXPECT_EQ(cache.stats().misses, 4U);
    EXPECT_LE(cache.stats().peak_bytes, 600U);
}

TEST(SchmidtRankCacheTest, AKeyLargerThanTheCapIsComputedButNotStored) {
    SchmidtRankCache cache(100);
    const std::vector<std::size_t> left{0};
    const std::vector<std::size_t> right{1};
    const std::vector<Complex> cx = gate(Opcode::CX);
    EXPECT_EQ(cache.rank(cx, 2, left, right), 2U);
    EXPECT_EQ(cache.rank(cx, 2, left, right), 2U);
    EXPECT_EQ(cache.stats().misses, 2U);
    EXPECT_EQ(cache.stats().uncached, 2U);
    EXPECT_EQ(cache.stats().bytes, 0U);
}

}  // namespace
}  // namespace menos::analysis::detail
