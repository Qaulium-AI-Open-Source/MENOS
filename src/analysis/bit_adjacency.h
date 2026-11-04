// Internal to the min-fill search (treewidth.cpp); exposed in this header
// only so tests/analysis/bit_adjacency_test.cpp and the planner structure
// microbenchmarks can exercise it alone (docs/adr/0025 section 12). Not public
// API.

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace menos::analysis::detail {

// Dense bitset adjacency for the min-fill search. The graph the search
// runs on (the closed network's line graph) is small -- hundreds to a few
// thousand nodes -- but DENSE once fill-in edges accumulate, which made the
// previous std::set<std::size_t>-per-vertex representation the planner's
// dominant cost (measured: ~96% of menos.run on a 10-qubit, 55-term QAOA
// estimate, almost all of it red-black-tree traversal). A row of 64-bit words
// per vertex turns "how many edges are there among v's neighbours" into
// popcounts of word-wise ANDs, and every neighbour set is still walked in
// ascending vertex order, so the elimination order, its tie-breaking (lowest
// index) and every reported width are unchanged -- only the data structure is.
class BitAdjacency {
public:
    // The words [first, last) of a row hold every one of its bits: first is
    // its lowest non-zero word and last is one past its highest. {0, 0} for an
    // empty row.
    struct WordRange {
        std::size_t first;
        std::size_t last;
    };

    explicit BitAdjacency(std::size_t n)
        : n_(n),
          words_((n + 63) / 64),
          bits_(n * words_, 0),
          span_(n, WordRange{0, 0}),
          degree_(n, 0) {}

    // Every row's WordRange is kept exact as bits change, so nonzero_range is
    // a lookup rather than a scan of the row (docs/adr/0025 section 12.3): a
    // set can only widen it, and a reset that empties the word at either end
    // narrows it past every empty word there.
    // Sets bit b of a's row; the search keeps the graph symmetric by always
    // calling set(a, b) and set(b, a) together.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (row, column) of one bit
    void set(std::size_t a, std::size_t b) {
        const std::size_t w = b / 64;
        const std::uint64_t bit = std::uint64_t{1} << (b % 64);
        std::uint64_t& word = row(a)[w];
        degree_[a] += (word & bit) == 0 ? 1U : 0U;
        word |= bit;
        WordRange& span = span_[a];
        if (span.first == span.last) {
            span = {w, w + 1};
        } else {
            span.first = std::min(span.first, w);
            span.last = std::max(span.last, w + 1);
        }
    }
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (row, column) of one bit
    void reset(std::size_t a, std::size_t b) {
        const std::size_t w = b / 64;
        std::uint64_t* r = row(a);
        const std::uint64_t bit = std::uint64_t{1} << (b % 64);
        degree_[a] -= (r[w] & bit) != 0 ? 1U : 0U;
        r[w] &= ~bit;
        WordRange& span = span_[a];
        if (r[w] != 0 || (w != span.first && w + 1 != span.last)) {
            return;
        }
        while (span.first < span.last && r[span.first] == 0) {
            ++span.first;
        }
        while (span.last > span.first && r[span.last - 1] == 0) {
            --span.last;
        }
        if (span.first == span.last) {
            span = {0, 0};
        }
    }
    void clear_row(std::size_t a) {
        const WordRange span = span_[a];
        std::fill(row(a) + span.first, row(a) + span.last, std::uint64_t{0});
        span_[a] = {0, 0};
        degree_[a] = 0;
    }

    // Kept as a count by set, reset and clear_row.
    [[nodiscard]] std::size_t degree(std::size_t a) const {
        return degree_[a];
    }
    // a's row: words_per_row() 64-bit words, bit u of the row set iff u is a
    // neighbour of a.
    [[nodiscard]] const std::uint64_t* row_words(std::size_t a) const {
        return row(a);
    }
    [[nodiscard]] bool test(std::size_t a, std::size_t b) const {
        return ((row(a)[b / 64] >> (b % 64)) & std::uint64_t{1}) != 0;
    }
    [[nodiscard]] WordRange nonzero_range(std::size_t a) const {
        return span_[a];
    }
    // |N(a) & N(b)|, given `b_range`, the non-zero range of b's row
    // (nonzero_range). Every word of b's row outside it is zero, so its AND
    // with a's row is zero too: summing over the range alone gives the same
    // integer as summing over the whole row, in time proportional to the span
    // of b's neighbours rather than to the graph's size.
    [[nodiscard]] std::size_t common(std::size_t a, std::size_t b, WordRange b_range) const {
        std::size_t c = 0;
        const std::uint64_t* ra = row(a);
        const std::uint64_t* rb = row(b);
        for (std::size_t w = b_range.first; w < b_range.last; ++w) {
            c += static_cast<std::size_t>(std::popcount(ra[w] & rb[w]));
        }
        return c;
    }
    // Calls f(u) for every neighbour u of a, in ascending order of u, given
    // a's non-zero range.
    template <typename F>
    void for_each_neighbor_in(std::size_t a, WordRange range, F&& f) const {
        const std::uint64_t* r = row(a);
        for (std::size_t w = range.first; w < range.last; ++w) {
            std::uint64_t word = r[w];
            while (word != 0) {
                const auto bit = static_cast<std::size_t>(std::countr_zero(word));
                f(w * 64 + bit);
                word &= word - 1;
            }
        }
    }
    // Calls f(u) for every common neighbour u of a and b, in ascending order
    // of u.
    template <typename F>
    void for_each_common_neighbor(std::size_t a, std::size_t b, F&& f) const {
        const WordRange sa = span_[a];
        const WordRange sb = span_[b];
        const std::size_t first = std::max(sa.first, sb.first);
        const std::size_t last = std::min(sa.last, sb.last);
        const std::uint64_t* ra = row(a);
        const std::uint64_t* rb = row(b);
        for (std::size_t w = first; w < last; ++w) {
            std::uint64_t word = ra[w] & rb[w];
            while (word != 0) {
                const auto bit = static_cast<std::size_t>(std::countr_zero(word));
                f(w * 64 + bit);
                word &= word - 1;
            }
        }
    }
    // Calls f(u) for every neighbour u of a, in ascending order of u.
    template <typename F>
    void for_each_neighbor(std::size_t a, F&& f) const {
        for_each_neighbor_in(a, span_[a], std::forward<F>(f));
    }
    [[nodiscard]] std::size_t num_nodes() const {
        return n_;
    }

private:
    std::uint64_t* row(std::size_t a) {
        return bits_.data() + a * words_;
    }
    [[nodiscard]] const std::uint64_t* row(std::size_t a) const {
        return bits_.data() + a * words_;
    }

    std::size_t n_;
    std::size_t words_;
    std::vector<std::uint64_t> bits_;
    std::vector<WordRange> span_;
    std::vector<std::size_t> degree_;
};

// fill(v) = C(deg(v), 2) - |{edges already present among v's neighbors}|
// -- the number of NEW fill-in edges eliminating v right now would add.
// Defined in treewidth.cpp, where its CPU-dispatch attribute lives.
// `scratch` is reused between calls to hold v's neighbours; its contents on
// entry and exit are unspecified.
[[nodiscard]] std::size_t compute_fill(const BitAdjacency& adjacency,
                                       std::size_t v,
                                       std::vector<std::size_t>& scratch);

}  // namespace menos::analysis::detail
