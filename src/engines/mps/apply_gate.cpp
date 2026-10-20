#include "engines/mps/apply_gate.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "common/memory_ledger.h"
#include "common/svd.h"
#include "ir/gate_table.h"

namespace menos::engines::mps {
namespace {

using Complex = std::complex<double>;

// A 1-qubit unitary preserves whichever isometry property its site
// already has, regardless of where the orthogonality center currently
// is (a direct consequence of the gate being unitary: contracting a
// unitary into an isometric tensor's physical index yields another
// isometric tensor -- see docs/adr/0009 or the derivation in the
// commit message). So this never moves the center, never touches any
// other site, and never truncates: no bond dimension is involved.
void apply_single_qubit_gate(MPSState& state, QubitId qubit, std::span<const Complex> matrix) {
    SiteTensor& site = state.site(qubit);
    SiteTensor result{site.chi_left,
                      site.chi_right,
                      std::vector<Complex>(site.chi_left * 2 * site.chi_right, Complex{0.0, 0.0})};
    for (std::size_t l = 0; l < site.chi_left; ++l) {
        for (std::size_t s_out = 0; s_out < 2; ++s_out) {
            for (std::size_t r = 0; r < site.chi_right; ++r) {
                Complex sum{0.0, 0.0};
                for (std::size_t s_in = 0; s_in < 2; ++s_in) {
                    sum += matrix[s_out * 2 + s_in] * site.at(l, s_in, r);
                }
                result.at(l, s_out, r) = sum;
            }
        }
    }
    site = std::move(result);
}

// Permutes matrix's bit order from "bit o <-> operand o" (CLAUDE.md's
// operand-order rule) to the block-local convention apply_local_block_gate
// needs: bit (k - 1 - rank[o]) <-> operand o, where rank[o] is operand
// o's position among the operands sorted by ascending qubit id (so
// operand rank 0, the smallest qubit, becomes the MOST significant
// local bit) -- this MSB-first-by-site-order layout is what makes each
// of apply_local_block_gate's sequential SVD splits a plain reshape of
// its running tensor's own flat storage, with no repacking (see that
// function's own comment).
std::vector<Complex> localize_matrix(std::span<const Complex> matrix,
                                     std::size_t k,
                                     const std::vector<std::size_t>& rank) {
    const std::size_t dim = std::size_t{1} << k;
    std::vector<std::size_t> dest_bit(k);
    for (std::size_t o = 0; o < k; ++o) {
        dest_bit[o] = k - 1 - rank[o];
    }
    std::vector<Complex> result(dim * dim);
    for (std::size_t row_out = 0; row_out < dim; ++row_out) {
        for (std::size_t col_out = 0; col_out < dim; ++col_out) {
            std::size_t row_in = 0;
            std::size_t col_in = 0;
            for (std::size_t o = 0; o < k; ++o) {
                if ((row_out >> dest_bit[o]) & std::size_t{1}) {
                    row_in |= std::size_t{1} << o;
                }
                if ((col_out >> dest_bit[o]) & std::size_t{1}) {
                    col_in |= std::size_t{1} << o;
                }
            }
            result[row_out * dim + col_out] = matrix[row_in * dim + col_in];
        }
    }
    return result;
}

// Keeps singular_values[0 .. kept) under policy, recording whatever is
// dropped into stats. Exact keeps every value (kept == full rank,
// matching MPSState::move_orthogonality_center_to's own losslessness).
// Controlled keeps sigma_i while sigma_i > tolerance * sigma_max (the
// same relative-cutoff convention common::numerical_rank uses, with
// policy.tolerance in place of machine epsilon), then additionally
// caps the count at max_bond_dimension when set. Always keeps at least
// 1 (a 0-dimensional bond is not a representable state).
std::size_t choose_kept_count(const std::vector<double>& singular_values,
                              const AccuracyPolicy& policy,
                              TruncationStats& stats) {
    const std::size_t full = singular_values.size();
    if (policy.kind == AccuracyPolicyKind::Exact || full == 0) {
        return full;
    }

    const double cutoff = policy.tolerance * singular_values.front();
    std::size_t kept = 0;
    while (kept < full && singular_values[kept] > cutoff) {
        ++kept;
    }
    if (policy.max_bond_dimension.has_value()) {
        kept = std::min(kept, static_cast<std::size_t>(*policy.max_bond_dimension));
    }
    kept = std::max<std::size_t>(kept, 1);

    if (kept < full) {
        double discarded_weight = 0.0;
        for (std::size_t i = kept; i < full; ++i) {
            discarded_weight += singular_values[i] * singular_values[i];
        }
        stats.discarded_weight_per_step.push_back(discarded_weight);
    }
    return kept;
}

// Applies the k-qubit matrix_local (already in the block-local MSB-
// first convention localize_matrix produces) to the k consecutive
// sites [site_start, site_start + k). Moves the center to site_start,
// contracts those k tensors into one (chi_left x 2^k x chi_right)
// tensor (combo's bit (k - 1 - p) is site (site_start + p)'s physical
// index, by construction below), applies matrix_local, then peels off
// k - 1 left-isometric site tensors via sequential SVD splits -- each
// split's own reshape is a plain reinterpretation of the running
// remainder's flat storage (no data movement) because of the MSB-first
// convention: peeling the current MSB groups (left bond, that bit) as
// the matrix's row space and (remaining bits, right bond) as its
// column space, which is exactly how the remainder is already laid
// out in memory. The final remainder (1 physical bit left) becomes
// site (site_start + k - 1) directly and is the new orthogonality
// center.
void apply_local_block_gate(MPSState& state,
                            std::size_t site_start,
                            std::size_t k,
                            std::span<const Complex> matrix_local,
                            const AccuracyPolicy& policy,
                            TruncationStats& stats) {
    state.move_orthogonality_center_to(site_start);

    const std::size_t chi_left = state.site(site_start).chi_left;
    const std::size_t dim = std::size_t{1} << k;

    std::vector<Complex> theta;
    // The block's contracted tensors, beside the sites they came from.
    common::MemoryLedger::Charge block_charge;
    std::size_t combos = 2;
    std::size_t bond{};
    {
        const SiteTensor& first = state.site(site_start);
        bond = first.chi_right;
        theta.assign(chi_left * combos * bond, Complex{0.0, 0.0});
        for (std::size_t l = 0; l < chi_left; ++l) {
            for (std::size_t s = 0; s < 2; ++s) {
                for (std::size_t r = 0; r < bond; ++r) {
                    theta[(l * combos + s) * bond + r] = first.at(l, s, r);
                }
            }
        }
    }
    for (std::size_t step = 1; step < k; ++step) {
        const SiteTensor& next = state.site(site_start + step);
        const std::size_t new_combos = combos * 2;
        const std::size_t new_bond = next.chi_right;
        std::vector<Complex> new_theta(chi_left * new_combos * new_bond, Complex{0.0, 0.0});
        block_charge.set(common::buffer_bytes(theta) + common::buffer_bytes(new_theta));
        for (std::size_t l = 0; l < chi_left; ++l) {
            for (std::size_t combo_old = 0; combo_old < combos; ++combo_old) {
                for (std::size_t m = 0; m < bond; ++m) {
                    const Complex v = theta[(l * combos + combo_old) * bond + m];
                    for (std::size_t s = 0; s < 2; ++s) {
                        const std::size_t combo_new = combo_old * 2 + s;
                        for (std::size_t r = 0; r < new_bond; ++r) {
                            new_theta[(l * new_combos + combo_new) * new_bond + r] +=
                                v * next.at(m, s, r);
                        }
                    }
                }
            }
        }
        theta = std::move(new_theta);
        combos = new_combos;
        bond = new_bond;
    }
    const std::size_t chi_right_last = bond;

    std::vector<Complex> theta2(chi_left * dim * chi_right_last, Complex{0.0, 0.0});
    block_charge.set(common::buffer_bytes(theta) + common::buffer_bytes(theta2));
    for (std::size_t l = 0; l < chi_left; ++l) {
        for (std::size_t combo_out = 0; combo_out < dim; ++combo_out) {
            for (std::size_t r = 0; r < chi_right_last; ++r) {
                Complex sum{0.0, 0.0};
                for (std::size_t combo_in = 0; combo_in < dim; ++combo_in) {
                    sum += matrix_local[combo_out * dim + combo_in] *
                           theta[(l * dim + combo_in) * chi_right_last + r];
                }
                theta2[(l * dim + combo_out) * chi_right_last + r] = sum;
            }
        }
    }

    std::vector<Complex> remainder = std::move(theta2);
    std::size_t cur_left = chi_left;
    std::size_t remaining_bits = k;
    for (std::size_t j = 0; j + 1 < k; ++j) {
        const std::size_t m = cur_left * 2;
        const std::size_t n = (std::size_t{1} << (remaining_bits - 1)) * chi_right_last;
        const common::SvdResult svd_result = common::svd(remainder, m, n);
        const std::size_t full_rank = svd_result.singular_values.size();
        const std::size_t kept = choose_kept_count(svd_result.singular_values, policy, stats);

        SiteTensor new_site{cur_left, kept, std::vector<Complex>(m * kept, Complex{0.0, 0.0})};
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < kept; ++col) {
                new_site.data[row * kept + col] = svd_result.u[row * full_rank + col];
            }
        }
        state.site(site_start + j) = std::move(new_site);

        // Renormalize the kept singular values so the state stays unit
        // norm after a truncating split (standard TEBD/MPS practice --
        // the per-step discarded weight tracked in stats is the raw
        // material for the accumulated error bound; the state itself
        // must not be left under-normalized by it). Under Exact, kept
        // == full_rank and this factor is 1 up to floating-point
        // rounding, since a unitary gate leaves the pre-split state's
        // total weight at 1 already.
        double kept_weight = 0.0;
        for (std::size_t row = 0; row < kept; ++row) {
            kept_weight += svd_result.singular_values[row] * svd_result.singular_values[row];
        }
        const double rescale = 1.0 / std::sqrt(kept_weight);

        std::vector<Complex> next_remainder(kept * n, Complex{0.0, 0.0});
        block_charge.set(common::buffer_bytes(theta) + common::buffer_bytes(remainder) +
                         common::buffer_bytes(next_remainder) + common::buffer_bytes(svd_result.u) +
                         common::buffer_bytes(svd_result.vt));
        for (std::size_t row = 0; row < kept; ++row) {
            const Complex s = Complex{svd_result.singular_values[row] * rescale, 0.0};
            for (std::size_t col = 0; col < n; ++col) {
                next_remainder[row * n + col] = s * svd_result.vt[row * n + col];
            }
        }
        remainder = std::move(next_remainder);
        block_charge.set(common::buffer_bytes(theta) + common::buffer_bytes(remainder));
        cur_left = kept;
        --remaining_bits;
    }

    SiteTensor last{cur_left, chi_right_last, std::move(remainder)};
    state.site(site_start + k - 1) = std::move(last);
    state.set_orthogonality_center_unchecked(site_start + k - 1);
}

// Gathers the operands named by sorted_qubits (ascending) into the
// contiguous block [span_start, span_start + span_len) by adjacent
// SWAPs, preserving their relative (sorted) order, recording each
// swap's left site into swap_log so the caller can undo it afterward
// by replaying swap_log in reverse (SWAP is a self-inverse
// involution). A stable "bubble the marked items to the front" pass:
// occupant[p] tracks which qubit currently sits at site
// (span_start + p); scanning left to right, whenever the occupant at
// p is one of our operands, it is walked left (one adjacent SWAP per
// step) to the next open front slot. span_start and span_len are read
// directly from sorted_qubits by this function's one caller and never
// separately supplied, so a swap between them is not a realistic
// mistake at that call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void gather_qubits(MPSState& state,
                   const std::vector<QubitId>& sorted_qubits,
                   std::size_t span_start,
                   std::size_t span_len,
                   std::span<const Complex> swap_local,
                   const AccuracyPolicy& policy,
                   TruncationStats& stats,
                   std::vector<std::size_t>& swap_log) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::vector<QubitId> occupant(span_len);
    for (std::size_t p = 0; p < span_len; ++p) {
        occupant[p] = static_cast<QubitId>(span_start + p);
    }

    std::size_t insert_at = 0;
    for (std::size_t p = 0; p < span_len; ++p) {
        if (!std::binary_search(sorted_qubits.begin(), sorted_qubits.end(), occupant[p])) {
            continue;
        }
        for (std::size_t pos = p; pos > insert_at; --pos) {
            const std::size_t site_a = span_start + pos - 1;
            apply_local_block_gate(state, site_a, 2, swap_local, policy, stats);
            swap_log.push_back(site_a);
            std::swap(occupant[pos - 1], occupant[pos]);
        }
        ++insert_at;
    }
}

}  // namespace

void apply_gate(MPSState& state,
                std::span<const Complex> matrix,
                std::span<const QubitId> qubits,
                const AccuracyPolicy& policy,
                TruncationStats& stats) {
    const std::size_t k = qubits.size();
    if (k == 1) {
        apply_single_qubit_gate(state, qubits[0], matrix);
        return;
    }

    std::vector<QubitId> sorted_qubits(qubits.begin(), qubits.end());
    std::sort(sorted_qubits.begin(), sorted_qubits.end());
    std::vector<std::size_t> rank(k);
    for (std::size_t o = 0; o < k; ++o) {
        rank[o] = static_cast<std::size_t>(
            std::lower_bound(sorted_qubits.begin(), sorted_qubits.end(), qubits[o]) -
            sorted_qubits.begin());
    }
    const std::vector<Complex> matrix_local = localize_matrix(matrix, k, rank);

    const std::size_t span_start = sorted_qubits.front();
    const std::size_t span_len = sorted_qubits.back() - span_start + 1;

    const std::vector<Complex> swap_matrix = ir::generate_matrix(Opcode::SWAP, {});
    const std::vector<Complex> swap_local = localize_matrix(swap_matrix, 2, {0, 1});

    std::vector<std::size_t> swap_log;
    gather_qubits(state, sorted_qubits, span_start, span_len, swap_local, policy, stats, swap_log);

    apply_local_block_gate(state, span_start, k, matrix_local, policy, stats);

    for (auto it = swap_log.rbegin(); it != swap_log.rend(); ++it) {
        apply_local_block_gate(state, *it, 2, swap_local, policy, stats);
    }
}

}  // namespace menos::engines::mps
