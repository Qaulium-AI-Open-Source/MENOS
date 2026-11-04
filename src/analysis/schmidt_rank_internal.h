// Internal to compute_schmidt_rank_bound (schmidt_rank_bound.cpp): the
// operator-Schmidt-rank computation and the per-call rank cache. Exposed in
// this header only so tests/analysis/schmidt_rank_cache_test.cpp and the
// planner structure microbenchmarks can exercise them alone (docs/adr/0025
// section 12). Not public API.

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace menos::analysis::detail {

using Complex = std::complex<double>;

// The operator Schmidt rank of the k-qubit row-major matrix u across the
// split of its operand positions into left_positions and right_positions,
// by SVD (definition in schmidt_rank_bound.cpp).
[[nodiscard]] std::uint64_t operator_schmidt_rank(std::span<const Complex> u,
                                                  std::size_t k,
                                                  const std::vector<std::size_t>& left_positions,
                                                  const std::vector<std::size_t>& right_positions);

struct SchmidtRankCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;      // every rank computed, cached or not
    std::uint64_t evictions = 0;   // times the cache was cleared to stay under its cap
    std::uint64_t uncached = 0;    // misses whose key alone exceeded the cap
    std::uint64_t bytes = 0;       // key bytes currently stored
    std::uint64_t peak_bytes = 0;  // the most key bytes stored at once
};

// Operator Schmidt ranks already computed (docs/adr/0025 sections 10.4, 12.3).
//
// Key: four-byte fields -- the operand count k, the number of operands left of
// the cut, and each of those operand positions, ascending -- followed by the
// matrix's 16 * 4^k raw bytes. Every field has a fixed width and the matrix
// length is fixed by k, so two different (k, split, matrix) triples never
// produce the same key. Keying on bytes, not values, is conservative: equal
// matrices that differ in representation (0.0 and -0.0) are computed
// separately.
//
// The rank is an integer computed deterministically from exactly those
// inputs, so a hit returns the integer operator_schmidt_rank would return
// again: what the cache holds, and when it is emptied, can change only how
// often the SVD runs, never a result.
//
// Size: at most cap_bytes of stored key bytes (default 16 MiB). An insertion
// that would exceed the cap first clears the cache (simple, deterministic,
// and in the measured corpus never reached: the largest peak was 1 KiB). A
// key larger than the cap on its own -- a k >= 10 UnitaryOp -- is computed
// and not stored. One cache lives for one compute_schmidt_rank_bound call and
// is not shared between threads.
class SchmidtRankCache {
public:
    static constexpr std::size_t kDefaultCapBytes = std::size_t{16} << 20;

    explicit SchmidtRankCache(std::size_t cap_bytes = kDefaultCapBytes) : cap_bytes_(cap_bytes) {}

    [[nodiscard]] std::uint64_t rank(std::span<const Complex> u,
                                     std::size_t k,
                                     const std::vector<std::size_t>& left_positions,
                                     const std::vector<std::size_t>& right_positions);

    [[nodiscard]] const SchmidtRankCacheStats& stats() const {
        return stats_;
    }

private:
    std::size_t cap_bytes_;
    std::unordered_map<std::string, std::uint64_t> ranks_;
    std::string key_;
    SchmidtRankCacheStats stats_;
};

}  // namespace menos::analysis::detail
