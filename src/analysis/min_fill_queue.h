// Internal to the min-fill search (treewidth.cpp): which remaining vertex to
// eliminate next. Exposed in this header only so
// tests/analysis/min_fill_queue_test.cpp can compare it against a linear scan
// and the planner structure microbenchmarks can time it alone
// (docs/adr/0025 section 12). Not public API.

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace menos::analysis::detail {

// Holds every remaining vertex's current fill and answers "the remaining
// vertex with the smallest fill, ties to the lowest index" -- exactly the
// vertex a scan of every remaining vertex in increasing index order with a
// strict `<` would choose.
//
// A lazy-deletion min-heap of (fill, vertex) pairs: std::pair compares fill
// first, then vertex, which is that order. Entries are never changed in
// place. set_fill pushes a new entry only when a vertex's fill actually
// changes, and pop_min discards an entry whose vertex is already removed or
// whose fill is no longer that vertex's current fill. Every remaining vertex
// always has one entry holding its current fill, so the first entry that
// survives is the scan's choice.
class MinFillQueue {
public:
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

    explicit MinFillQueue(std::span<const std::size_t> initial_fill)
        : fill_(initial_fill.begin(), initial_fill.end()), removed_(initial_fill.size(), false) {
        std::vector<Entry> storage;
        storage.reserve(initial_fill.size());
        for (std::size_t v = 0; v < initial_fill.size(); ++v) {
            storage.emplace_back(initial_fill[v], v);
        }
        heap_ = Heap(std::greater<>{}, std::move(storage));
        peak_entries_ = heap_.size();
    }

    // Records v's current fill. v must not have been popped.
    void set_fill(std::size_t v, std::size_t fill) {
        if (fill == fill_[v]) {
            return;
        }
        fill_[v] = fill;
        heap_.emplace(fill, v);
        peak_entries_ = std::max(peak_entries_, heap_.size());
    }

    [[nodiscard]] std::size_t fill(std::size_t v) const {
        return fill_[v];
    }
    [[nodiscard]] bool removed(std::size_t v) const {
        return removed_[v];
    }

    // Removes and returns the remaining vertex with the smallest fill, ties
    // to the lowest index; kNone if every vertex has been removed.
    std::size_t pop_min() {
        while (!heap_.empty()) {
            const auto [entry_fill, v] = heap_.top();
            heap_.pop();
            if (!removed_[v] && entry_fill == fill_[v]) {
                removed_[v] = true;
                return v;
            }
        }
        return kNone;
    }

    // The most entries the heap has held at once, for the memory report
    // (docs/adr/0025 section 12.1, risk 1).
    [[nodiscard]] std::size_t peak_entries() const {
        return peak_entries_;
    }

private:
    using Entry = std::pair<std::size_t, std::size_t>;
    using Heap = std::priority_queue<Entry, std::vector<Entry>, std::greater<>>;

    std::vector<std::size_t> fill_;
    std::vector<bool> removed_;
    Heap heap_;
    std::size_t peak_entries_ = 0;
};

}  // namespace menos::analysis::detail
