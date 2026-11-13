// Targeted tests for the min-fill vertex choice (src/analysis/min_fill_queue.h;
// docs/adr/0028 sections 10.2 and 12).
//
// The oracle is the scan the queue replaced: over every remaining vertex in
// increasing index order, replace the best only on a strictly smaller fill,
// so ties go to the lowest index. Fills are drawn from a handful of values,
// so most choices are ties; making ties go to any other vertex fails these
// tests (docs/adr/0028 section 12.2 records that mutation check).

#include "analysis/min_fill_queue.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace menos::analysis::detail {
namespace {

struct Oracle {
    std::vector<std::size_t> fill;
    std::vector<bool> removed;

    std::size_t pop_min() {
        std::size_t best = MinFillQueue::kNone;
        for (std::size_t v = 0; v < fill.size(); ++v) {
            if (!removed[v] && (best == MinFillQueue::kNone || fill[v] < fill[best])) {
                best = v;
            }
        }
        if (best != MinFillQueue::kNone) {
            removed[best] = true;
        }
        return best;
    }
};

// One random run: n vertices, fills in [0, max_fill], and between pops a
// random number of fill changes to random remaining vertices (including
// "changes" to the same value, and a change back to an earlier value).
struct Run {
    std::size_t n;
    std::size_t max_fill;
    std::uint64_t seed;
};

void run_against_oracle(Run run) {
    const std::size_t n = run.n;
    const std::size_t max_fill = run.max_fill;
    const std::uint64_t seed = run.seed;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> fill_value(0, max_fill);
    std::vector<std::size_t> initial(n);
    for (std::size_t& f : initial) {
        f = fill_value(rng);
    }
    MinFillQueue queue(initial);
    Oracle oracle{initial, std::vector<bool>(n, false)};
    std::uniform_int_distribution<std::size_t> vertex(0, n == 0 ? 0 : n - 1);
    std::uniform_int_distribution<std::size_t> updates(0, 2 * n);
    for (std::size_t step = 0; step <= n; ++step) {
        const std::size_t expected = oracle.pop_min();
        ASSERT_EQ(queue.pop_min(), expected) << "seed " << seed << " step " << step;
        if (expected == MinFillQueue::kNone) {
            break;
        }
        const std::size_t changes = updates(rng);
        for (std::size_t c = 0; c < changes; ++c) {
            const std::size_t v = vertex(rng);
            if (oracle.removed[v]) {
                ASSERT_TRUE(queue.removed(v));
                continue;
            }
            const std::size_t f = fill_value(rng);
            oracle.fill[v] = f;
            queue.set_fill(v, f);
            ASSERT_EQ(queue.fill(v), f);
        }
    }
    EXPECT_EQ(queue.pop_min(), MinFillQueue::kNone);
}

TEST(MinFillQueueTest, PicksExactlyWhatTheLinearScanPicksWithHeavyTies) {
    for (std::uint64_t seed = 0; seed < 400; ++seed) {
        run_against_oracle({1 + seed % 70, seed % 4, seed});
    }
}

TEST(MinFillQueueTest, PicksExactlyWhatTheLinearScanPicksWithWideFills) {
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        run_against_oracle({50 + seed, 1000, 1000 + seed});
    }
}

TEST(MinFillQueueTest, AllEqualFillsComeOutInIndexOrder) {
    MinFillQueue queue(std::vector<std::size_t>(130, 5));
    for (std::size_t v = 0; v < 130; ++v) {
        EXPECT_EQ(queue.pop_min(), v);
    }
    EXPECT_EQ(queue.pop_min(), MinFillQueue::kNone);
}

TEST(MinFillQueueTest, AFillThatChangesAndChangesBackKeepsItsPlace) {
    // Vertex 3's fill goes 1 -> 9 -> 1, leaving two entries (1, 3) in the
    // heap; it must come out once, and in index order among the fill-1 ties.
    MinFillQueue queue(std::vector<std::size_t>{2, 1, 2, 1, 1});
    queue.set_fill(3, 9);
    queue.set_fill(3, 1);
    EXPECT_EQ(queue.pop_min(), 1U);
    EXPECT_EQ(queue.pop_min(), 3U);
    EXPECT_EQ(queue.pop_min(), 4U);
    EXPECT_EQ(queue.pop_min(), 0U);
    EXPECT_EQ(queue.pop_min(), 2U);
    EXPECT_EQ(queue.pop_min(), MinFillQueue::kNone);
}

TEST(MinFillQueueTest, AnEmptyQueueHasNothingToPop) {
    MinFillQueue queue(std::vector<std::size_t>{});
    EXPECT_EQ(queue.pop_min(), MinFillQueue::kNone);
}

}  // namespace
}  // namespace menos::analysis::detail
