// Internal to the min-fill search (treewidth.cpp): which remaining vertex to
// eliminate next. Exposed in this header only so
// tests/analysis/min_fill_queue_test.cpp can compare it against a linear scan
// and the planner structure microbenchmarks can time it alone
// (docs/adr/0025 section 12). Not public API.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "common/error.h"

namespace menos::analysis::detail {

// Holds every remaining vertex's current fill and answers "the remaining
// vertex with the smallest fill, ties to the lowest index" -- exactly the
// vertex a scan of every remaining vertex in increasing index order with a
// strict `<` would choose.
//
// An indexed binary min-heap (docs/adr/0025 section 12.3, C.2a): one entry
// per remaining vertex, and a position table so set_fill moves a vertex's
// entry up or down in place instead of adding a new one. The heap therefore
// never holds more than n entries. Each entry is one integer,
// (fill << 32) | vertex, so the integer order is fill first, then vertex --
// the scan's order -- and a comparison is a single integer compare. The
// packing needs vertex and fill below 2^32; the constructor checks both
// bounds, which the dense search's size limit (estimate_treewidth_min_fill)
// keeps far from: a fill is at most C(n - 1, 2).
class MinFillQueue {
public:
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

    explicit MinFillQueue(std::span<const std::size_t> initial_fill)
        : fill_(initial_fill.begin(), initial_fill.end()),
          position_(initial_fill.size(), kRemoved) {
        const std::size_t n = initial_fill.size();
        common::ensure(n <= kLow && (n < 2 || (n - 1) * (n - 2) / 2 <= kLow),
                       "MinFillQueue: graph too large for packed heap keys");
        heap_.reserve(n);
        for (std::size_t v = 0; v < n; ++v) {
            heap_.push_back(key(v));
            position_[v] = v;
        }
        // Floyd's bottom-up heap construction.
        for (std::size_t i = n / 2; i-- > 0;) {
            sift_down(i);
        }
    }

    // Records v's current fill. v must not have been popped.
    void set_fill(std::size_t v, std::size_t fill) {
        const std::size_t old = fill_[v];
        if (fill == old) {
            return;
        }
        fill_[v] = fill;
        const std::size_t at = position_[v];
        heap_[at] = key(v);
        if (fill < old) {
            sift_up(at);
        } else {
            sift_down(at);
        }
    }

    [[nodiscard]] std::size_t fill(std::size_t v) const {
        return fill_[v];
    }
    [[nodiscard]] bool removed(std::size_t v) const {
        return position_[v] == kRemoved;
    }

    // Removes and returns the remaining vertex with the smallest fill, ties
    // to the lowest index; kNone if every vertex has been removed.
    std::size_t pop_min() {
        if (heap_.empty()) {
            return kNone;
        }
        const auto v = static_cast<std::size_t>(heap_.front() & kLow);
        position_[v] = kRemoved;
        const std::uint64_t last = heap_.back();
        heap_.pop_back();
        if (!heap_.empty()) {
            heap_.front() = last;
            position_[static_cast<std::size_t>(last & kLow)] = 0;
            sift_down(0);
        }
        return v;
    }

    // The most entries the heap has held at once: n, by construction.
    [[nodiscard]] std::size_t peak_entries() const {
        return fill_.size();
    }

private:
    static constexpr std::uint64_t kLow = 0xFFFFFFFFULL;
    static constexpr std::size_t kRemoved = std::numeric_limits<std::size_t>::max();

    [[nodiscard]] std::uint64_t key(std::size_t v) const {
        return (static_cast<std::uint64_t>(fill_[v]) << 32) | static_cast<std::uint64_t>(v);
    }
    void place(std::size_t at, std::uint64_t k) {
        heap_[at] = k;
        position_[static_cast<std::size_t>(k & kLow)] = at;
    }
    void sift_up(std::size_t at) {
        const std::uint64_t k = heap_[at];
        while (at > 0) {
            const std::size_t parent = (at - 1) / 2;
            if (heap_[parent] <= k) {
                break;
            }
            place(at, heap_[parent]);
            at = parent;
        }
        place(at, k);
    }
    void sift_down(std::size_t at) {
        const std::size_t n = heap_.size();
        const std::uint64_t k = heap_[at];
        while (true) {
            std::size_t child = 2 * at + 1;
            if (child >= n) {
                break;
            }
            if (child + 1 < n && heap_[child + 1] < heap_[child]) {
                ++child;
            }
            if (k <= heap_[child]) {
                break;
            }
            place(at, heap_[child]);
            at = child;
        }
        place(at, k);
    }

    std::vector<std::size_t> fill_;
    std::vector<std::size_t> position_;  // index into heap_, or kRemoved
    std::vector<std::uint64_t> heap_;
};

}  // namespace menos::analysis::detail
