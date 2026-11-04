// Internal to the min-fill search (treewidth.cpp); exposed in this header
// only so tests/analysis/bit_adjacency_test.cpp and the planner structure
// microbenchmarks can exercise it alone (docs/adr/0025 section 12). Not public
// API.

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
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
    explicit BitAdjacency(std::size_t n) : n_(n), words_((n + 63) / 64), bits_(n * words_, 0) {}

    void set(std::size_t a, std::size_t b) {
        row(a)[b / 64] |= std::uint64_t{1} << (b % 64);
    }
    void reset(std::size_t a, std::size_t b) {
        row(a)[b / 64] &= ~(std::uint64_t{1} << (b % 64));
    }
    void clear_row(std::size_t a) {
        std::fill_n(row(a), words_, std::uint64_t{0});
    }

    [[nodiscard]] std::size_t degree(std::size_t a) const {
        std::size_t d = 0;
        const std::uint64_t* r = row(a);
        for (std::size_t w = 0; w < words_; ++w) {
            d += static_cast<std::size_t>(std::popcount(r[w]));
        }
        return d;
    }
    // The words [first, last) of a's row hold every one of its bits: first is
    // its lowest non-zero word and last is one past its highest. {0, 0} for an
    // empty row.
    struct WordRange {
        std::size_t first;
        std::size_t last;
    };
    [[nodiscard]] WordRange nonzero_range(std::size_t a) const {
        const std::uint64_t* r = row(a);
        std::size_t first = 0;
        while (first < words_ && r[first] == 0) {
            ++first;
        }
        if (first == words_) {
            return {0, 0};
        }
        std::size_t last = words_;
        while (r[last - 1] == 0) {
            --last;
        }
        return {first, last};
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
    // Calls f(u) for every neighbour u of a, in ascending order of u.
    template <typename F>
    void for_each_neighbor(std::size_t a, F&& f) const {
        const std::uint64_t* r = row(a);
        for (std::size_t w = 0; w < words_; ++w) {
            std::uint64_t word = r[w];
            while (word != 0) {
                const auto bit = static_cast<std::size_t>(std::countr_zero(word));
                f(w * 64 + bit);
                word &= word - 1;
            }
        }
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
};

// fill(v) = C(deg(v), 2) - |{edges already present among v's neighbors}|
// -- the number of NEW fill-in edges eliminating v right now would add.
// Defined in treewidth.cpp, where its CPU-dispatch attribute lives.
[[nodiscard]] std::size_t compute_fill(const BitAdjacency& adjacency, std::size_t v);

}  // namespace menos::analysis::detail
