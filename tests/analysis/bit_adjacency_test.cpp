// Targeted tests for the min-fill search's bitset adjacency
// (src/analysis/bit_adjacency.h; docs/adr/0025 sections 10.1 and 12).
//
// common(a, b, range) sums popcounts over b's non-zero word span only; the
// oracle here is the plain popcount over every word of both rows, computed
// from an independent std::vector<bool> copy of the graph. Rows cover the
// empty, single-bit, sparse, half-full and full cases, with bits placed on
// and around 64-bit word boundaries, for sizes either side of each boundary.

#include "analysis/bit_adjacency.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace menos::analysis::detail {
namespace {

using Matrix = std::vector<std::vector<bool>>;

struct Graph {
    BitAdjacency bits;
    Matrix plain;
};

// A symmetric graph with no self-loops on n vertices; `density` is the
// probability of each edge, and `boundary_bits` additionally connects vertex 0
// to every vertex on either side of a word boundary.
struct Shape {
    std::size_t n;
    double density;
    bool boundary_bits;
};

Graph random_graph(Shape shape, std::mt19937_64& rng) {
    const std::size_t n = shape.n;
    const double density = shape.density;
    const bool boundary_bits = shape.boundary_bits;
    Graph g{BitAdjacency(n), Matrix(n, std::vector<bool>(n, false))};
    std::bernoulli_distribution edge(density);
    const auto connect = [&](std::size_t a, std::size_t b) {
        if (a != b && a < n && b < n) {
            g.bits.set(a, b);
            g.bits.set(b, a);
            g.plain[a][b] = true;
            g.plain[b][a] = true;
        }
    };
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a + 1; b < n; ++b) {
            if (edge(rng)) {
                connect(a, b);
            }
        }
    }
    if (boundary_bits) {
        for (std::size_t w = 64; w <= n; w += 64) {
            connect(0, w - 1);
            connect(0, w);
        }
    }
    return g;
}

std::size_t plain_common(const Matrix& m, std::size_t a, std::size_t b) {
    std::size_t c = 0;
    for (std::size_t u = 0; u < m.size(); ++u) {
        c += (m[a][u] && m[b][u]) ? 1U : 0U;
    }
    return c;
}

std::size_t plain_fill(const Matrix& m, std::size_t v) {
    std::vector<std::size_t> nb;
    for (std::size_t u = 0; u < m.size(); ++u) {
        if (m[v][u]) {
            nb.push_back(u);
        }
    }
    std::size_t missing = 0;
    for (std::size_t i = 0; i < nb.size(); ++i) {
        for (std::size_t j = i + 1; j < nb.size(); ++j) {
            missing += m[nb[i]][nb[j]] ? 0U : 1U;
        }
    }
    return missing;
}

void check_graph(const Graph& g) {
    const std::size_t n = g.plain.size();
    for (std::size_t a = 0; a < n; ++a) {
        const BitAdjacency::WordRange range = g.bits.nonzero_range(a);
        // The range is empty exactly for an empty row, and every neighbour
        // lies inside it, visited in ascending order.
        std::vector<std::size_t> expected;
        for (std::size_t u = 0; u < n; ++u) {
            if (g.plain[a][u]) {
                expected.push_back(u);
            }
        }
        std::vector<std::size_t> seen;
        g.bits.for_each_neighbor_in(a, range, [&](std::size_t u) { seen.push_back(u); });
        ASSERT_EQ(seen, expected) << "vertex " << a;
        ASSERT_EQ(g.bits.degree(a), expected.size());
        if (expected.empty()) {
            ASSERT_EQ(range.first, 0U);
            ASSERT_EQ(range.last, 0U);
        } else {
            ASSERT_EQ(range.first, expected.front() / 64);
            ASSERT_EQ(range.last, expected.back() / 64 + 1);
        }
        for (std::size_t b = 0; b < n; ++b) {
            const BitAdjacency::WordRange b_range = g.bits.nonzero_range(b);
            ASSERT_EQ(g.bits.common(a, b, b_range), plain_common(g.plain, a, b))
                << "a " << a << " b " << b << " n " << n;
        }
        ASSERT_EQ(compute_fill(g.bits, a), plain_fill(g.plain, a)) << "vertex " << a;
    }
}

TEST(BitAdjacencyTest, CommonEqualsTheFullPopcountAcrossDensitiesAndWordBoundaries) {
    std::mt19937_64 rng(20261104);
    for (const std::size_t n : {1U, 2U, 63U, 64U, 65U, 127U, 128U, 129U, 200U}) {
        for (const double density : {0.0, 0.01, 0.1, 0.5, 1.0}) {
            for (const bool boundary : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << "n " << n << " density " << density << " boundary " << boundary);
                check_graph(random_graph({n, density, boundary}, rng));
            }
        }
    }
}

TEST(BitAdjacencyTest, SingleBitRowsAtEveryPositionOfALargeGraph) {
    // n = 1000: 16 words, so the span of a single-bit row is one word far
    // from either end.
    constexpr std::size_t kN = 1000;
    for (const std::size_t u : {0U, 1U, 63U, 64U, 511U, 512U, 998U, 999U}) {
        BitAdjacency bits(kN);
        const std::size_t v = u == 500 ? 501 : 500;
        bits.set(v, u);
        bits.set(u, v);
        const BitAdjacency::WordRange range = bits.nonzero_range(v);
        EXPECT_EQ(range.first, u / 64);
        EXPECT_EQ(range.last, u / 64 + 1);
        EXPECT_EQ(bits.common(u, v, range), 0U);
        EXPECT_EQ(bits.common(v, v, range), 1U);
        EXPECT_EQ(compute_fill(bits, v), 0U);
    }
}

TEST(BitAdjacencyTest, AnEmptiedRowHasAnEmptyRangeAndNoFill) {
    std::mt19937_64 rng(7);
    Graph g = random_graph({130, 0.5, true}, rng);
    g.bits.clear_row(0);
    const BitAdjacency::WordRange range = g.bits.nonzero_range(0);
    EXPECT_EQ(range.first, 0U);
    EXPECT_EQ(range.last, 0U);
    EXPECT_EQ(compute_fill(g.bits, 0), 0U);
    for (std::size_t b = 1; b < 130; ++b) {
        EXPECT_EQ(g.bits.common(b, 0, range), 0U);
    }
}

TEST(BitAdjacencyTest, TheSpanStaysExactUnderInterleavedSetsResetsAndClears) {
    // The span is maintained incrementally (docs/adr/0025 section 12.3); it
    // must match the row's real first and last non-zero words after every
    // operation, including resets that empty an end word and rows emptied
    // one bit at a time.
    std::mt19937_64 rng(12);
    for (const std::size_t n : {1U, 64U, 65U, 130U, 300U}) {
        BitAdjacency bits(n);
        Matrix plain(n, std::vector<bool>(n, false));
        std::uniform_int_distribution<std::size_t> vertex(0, n - 1);
        std::uniform_int_distribution<int> action(0, 99);
        for (int i = 0; i < 20000; ++i) {
            const std::size_t a = vertex(rng);
            const std::size_t b = vertex(rng);
            const int what = action(rng);
            if (what < 45) {
                bits.set(a, b);
                plain[a][b] = true;
            } else if (what < 99) {
                bits.reset(a, b);
                plain[a][b] = false;
            } else {
                bits.clear_row(a);
                plain[a].assign(n, false);
            }
            std::size_t first = n;
            std::size_t last = 0;
            std::size_t degree = 0;
            for (std::size_t u = 0; u < n; ++u) {
                if (plain[a][u]) {
                    first = std::min(first, u / 64);
                    last = u / 64 + 1;
                    ++degree;
                }
            }
            const BitAdjacency::WordRange range = bits.nonzero_range(a);
            if (degree == 0) {
                ASSERT_EQ(range.first, 0U) << "n " << n << " op " << i;
                ASSERT_EQ(range.last, 0U) << "n " << n << " op " << i;
            } else {
                ASSERT_EQ(range.first, first) << "n " << n << " op " << i;
                ASSERT_EQ(range.last, last) << "n " << n << " op " << i;
            }
            ASSERT_EQ(bits.degree(a), degree);
        }
    }
}

TEST(BitAdjacencyTest, FullGraphHasNoFill) {
    std::mt19937_64 rng(11);
    const Graph g = random_graph({129, 1.0, false}, rng);
    for (std::size_t v = 0; v < 129; ++v) {
        EXPECT_EQ(compute_fill(g.bits, v), 0U);
        EXPECT_EQ(g.bits.common(v, (v + 1) % 129, g.bits.nonzero_range((v + 1) % 129)), 127U);
    }
}

}  // namespace
}  // namespace menos::analysis::detail
