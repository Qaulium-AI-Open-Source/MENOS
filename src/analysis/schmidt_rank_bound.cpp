#include "analysis/schmidt_rank_bound.h"

#include "analysis/planner_counters.h"
#include "analysis/schmidt_rank_internal.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/svd.h"
#include "ir/gate_table.h"

namespace menos::analysis {
namespace {

using detail::Complex;

// (qubits, matrix) for a GateOp/UnitaryOp; the matrix is always owned
// here (generate_matrix returns by value for GateOp; UnitaryOp's own
// matrix is copied) -- this pass runs once per job, not a hot loop, so
// the copy is not a performance concern.
struct OperandMatrix {
    const std::vector<QubitId>* qubits;
    std::vector<Complex> matrix;
};

// nullopt (empty optional-by-convention: qubits == nullptr) for
// MeasureOp, which has no matrix and (being single-qubit) can never
// cross a cut anyway. Exhaustive over Operation: a new alternative fails the
// static_assert until it is given its own rule (docs/adr/0016 section 7).
OperandMatrix operand_matrix_of(const Operation& op) {
    return std::visit(
        [](const auto& o) -> OperandMatrix {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp>) {
                return OperandMatrix{&o.qubits, ir::generate_matrix(o.opcode, o.params)};
            } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                return OperandMatrix{&o.qubits, o.matrix};
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                // The bound is a pure-state quantity. Its only caller,
                // MPSEngine::estimate, refuses a noisy region before calling it.
                throw common::InternalError(
                    "schmidt_rank_bound: a noise channel reached a pure-state bound");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return OperandMatrix{nullptr, {}};
            }
        },
        op);
}

// The operands of a gate or unitary; nullptr for a measurement. A noise
// channel throws exactly as operand_matrix_of does, whatever its size.
const std::vector<QubitId>* operands_of(const Operation& op) {
    return std::visit(
        [](const auto& o) -> const std::vector<QubitId>* {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return &o.qubits;
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                throw common::InternalError(
                    "schmidt_rank_bound: a noise channel reached a pure-state bound");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return nullptr;
            }
        },
        op);
}

// 2^min(left_size, right_size), or, if that would not fit a uint64_t,
// the largest representable value -- a cut this wide is already far
// beyond any budget a caller could satisfy, so the cap's own exact
// magnitude past 2^62 is immaterial (checked_pow2's own limit, per
// CLAUDE.md's "never shift by n >= 63").
std::uint64_t subsystem_dimension_cap(std::uint32_t left_size, std::uint32_t right_size) {
    const std::uint32_t smaller = std::min(left_size, right_size);
    if (smaller > 62) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return common::checked_pow2(smaller);
}

}  // namespace

namespace detail {

// The operator-Schmidt-rank of the k-qubit matrix u (row-major,
// operand-order bit convention) with respect to splitting its k
// operand positions into left_positions and right_positions: reshape
// u's (row, col) index pair, each k bits, into a
// 2^(2*l) x 2^(2*r) matrix m whose row groups a left-side (row, col)
// bit pair and whose column groups a right-side (row, col) bit pair
// (the standard operator-Schmidt reshaping: u as a vector in
// (H_L(x)H_R)(x)(H_L(x)H_R)* regrouped as (H_L(x)H_L*)(x)(H_R(x)H_R*)),
// then take m's ordinary matrix rank via SVD.
std::uint64_t operator_schmidt_rank(std::span<const Complex> u,
                                    std::size_t k,
                                    const std::vector<std::size_t>& left_positions,
                                    const std::vector<std::size_t>& right_positions) {
    const std::size_t l = left_positions.size();
    const std::size_t r = right_positions.size();
    const std::size_t dim = std::size_t{1} << k;
    const std::size_t m_rows = std::size_t{1} << (2 * l);
    const std::size_t m_cols = std::size_t{1} << (2 * r);
    std::vector<Complex> m(m_rows * m_cols);

    for (std::size_t i_l = 0; i_l < (std::size_t{1} << l); ++i_l) {
        for (std::size_t j_l = 0; j_l < (std::size_t{1} << l); ++j_l) {
            const std::size_t row_combo = i_l * (std::size_t{1} << l) + j_l;
            for (std::size_t i_r = 0; i_r < (std::size_t{1} << r); ++i_r) {
                for (std::size_t j_r = 0; j_r < (std::size_t{1} << r); ++j_r) {
                    std::size_t row = 0;
                    std::size_t col = 0;
                    for (std::size_t p = 0; p < l; ++p) {
                        if ((i_l >> p) & std::size_t{1}) {
                            row |= std::size_t{1} << left_positions[p];
                        }
                        if ((j_l >> p) & std::size_t{1}) {
                            col |= std::size_t{1} << left_positions[p];
                        }
                    }
                    for (std::size_t p = 0; p < r; ++p) {
                        if ((i_r >> p) & std::size_t{1}) {
                            row |= std::size_t{1} << right_positions[p];
                        }
                        if ((j_r >> p) & std::size_t{1}) {
                            col |= std::size_t{1} << right_positions[p];
                        }
                    }
                    const std::size_t col_combo = i_r * (std::size_t{1} << r) + j_r;
                    m[row_combo * m_cols + col_combo] = u[row * dim + col];
                }
            }
        }
    }

    const common::SvdResult svd_result = common::svd(m, m_rows, m_cols);
    return common::numerical_rank(svd_result.singular_values, m_rows, m_cols);
}

namespace {

void append_field(std::string& key, std::size_t value) {
    const auto field = static_cast<std::uint32_t>(value);
    common::ensure(field == value, "SchmidtRankCache: key field does not fit 32 bits");
    std::array<char, sizeof(field)> bytes{};
    std::memcpy(bytes.data(), &field, sizeof(field));
    key.append(bytes.data(), bytes.size());
}

}  // namespace

std::uint64_t SchmidtRankCache::rank(std::span<const Complex> u,
                                     std::size_t k,
                                     const std::vector<std::size_t>& left_positions,
                                     const std::vector<std::size_t>& right_positions) {
    key_.clear();
    append_field(key_, k);
    append_field(key_, left_positions.size());
    for (const std::size_t p : left_positions) {
        append_field(key_, p);
    }
    key_.append(reinterpret_cast<const char*>(u.data()), u.size() * sizeof(Complex));
    const auto cached = ranks_.find(key_);
    if (cached != ranks_.end()) {
        ++stats_.hits;
        return cached->second;
    }
    ++stats_.misses;
    const std::uint64_t rank = operator_schmidt_rank(u, k, left_positions, right_positions);
    if (key_.size() > cap_bytes_) {
        ++stats_.uncached;
        return rank;
    }
    if (stats_.bytes + key_.size() > cap_bytes_) {
        ranks_.clear();
        stats_.bytes = 0;
        ++stats_.evictions;
    }
    ranks_.emplace(key_, rank);
    stats_.bytes += key_.size();
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.bytes);
    return rank;
}

}  // namespace detail

std::uint64_t SchmidtRankBound::max_bound() const {
    if (bound_per_cut.empty()) {
        return 1;
    }
    return *std::max_element(bound_per_cut.begin(), bound_per_cut.end());
}

SchmidtRankBound compute_schmidt_rank_bound(std::uint32_t num_qubits,
                                            std::span<const Operation> operations) {
    const std::uint32_t n = num_qubits;
    SchmidtRankBound result;
    if (n < 2) {
        return result;
    }
    result.bound_per_cut.assign(n - 1, 1);

    std::vector<std::uint64_t> caps(n - 1);
    for (std::uint32_t i = 0; i < n - 1; ++i) {
        caps[i] = subsystem_dimension_cap(i + 1, n - (i + 1));
    }

    // Ranks already computed in this call (schmidt_rank_internal.h).
    detail::SchmidtRankCache rank_cache;
    for (const Operation& op : operations) {
        // A single-qubit operation or a measurement never crosses a cut, so
        // the operands are checked before any matrix is generated.
        const std::vector<QubitId>* operands = operands_of(op);
        if (operands == nullptr) {
            continue;  // MeasureOp: no matrix, never crosses a cut.
        }
        const QubitId min_q = *std::min_element(operands->begin(), operands->end());
        const QubitId max_q = *std::max_element(operands->begin(), operands->end());
        if (min_q == max_q) {
            continue;  // Single-qubit operation: never crosses a cut.
        }
        const OperandMatrix operand_matrix = operand_matrix_of(op);
        const std::vector<QubitId>& qubits = *operand_matrix.qubits;
        const std::size_t k = qubits.size();

        for (QubitId cut = min_q; cut < max_q; ++cut) {
            if (result.bound_per_cut[cut] >= caps[cut]) {
                continue;  // Already at the physical cap; further
                           // multiplication cannot raise it.
            }
            std::vector<std::size_t> left_positions;
            std::vector<std::size_t> right_positions;
            for (std::size_t p = 0; p < k; ++p) {
                if (qubits[p] <= cut) {
                    left_positions.push_back(p);
                } else {
                    right_positions.push_back(p);
                }
            }
            const std::uint64_t op_rank =
                rank_cache.rank(operand_matrix.matrix, k, left_positions, right_positions);
            result.bound_per_cut[cut] =
                std::min(common::checked_mul(result.bound_per_cut[cut], op_rank), caps[cut]);
        }
    }

#ifdef MENOS_PLANNER_COUNTERS
    detail::planner_counters.add(rank_cache.stats());
#endif
    return result;
}

}  // namespace menos::analysis
