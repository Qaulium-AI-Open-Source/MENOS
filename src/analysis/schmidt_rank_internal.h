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
    std::uint64_t misses = 0;
    std::uint64_t bytes = 0;       // key bytes currently stored
    std::uint64_t peak_bytes = 0;  // the most key bytes stored at once
};

// Operator Schmidt ranks already computed, keyed by which operands lie left
// of the cut and the exact bytes of the matrix. The rank is an integer
// computed deterministically from exactly those inputs, so a repeated key
// returns the integer operator_schmidt_rank would return again. Keying on
// bytes, not values, is conservative: equal matrices that differ in
// representation (0.0 and -0.0) are computed separately. One cache lives for
// one compute_schmidt_rank_bound call; it is not shared between threads.
class SchmidtRankCache {
public:
    [[nodiscard]] std::uint64_t rank(std::span<const Complex> u,
                                     std::size_t k,
                                     const std::vector<std::size_t>& left_positions,
                                     const std::vector<std::size_t>& right_positions);

    [[nodiscard]] const SchmidtRankCacheStats& stats() const {
        return stats_;
    }

private:
    std::unordered_map<std::string, std::uint64_t> ranks_;
    std::string key_;
    SchmidtRankCacheStats stats_;
};

}  // namespace menos::analysis::detail
