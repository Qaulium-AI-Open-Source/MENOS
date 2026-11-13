#include "engines/mps/mps_state.h"

#include <algorithm>

#include "common/error.h"
#include "common/memory_ledger.h"
#include "common/svd.h"

namespace menos::engines::mps {
namespace {

using Complex = std::complex<double>;

// Moves the center from p to p+1: SVDs site p (already exactly a
// (chi_left*2) x chi_right row-major matrix in its own storage layout,
// no repacking needed), keeps site p's U as the new (left-isometric)
// site p, and absorbs S*Vt into site p+1's left bond by explicit
// contraction (site p+1's own layout interleaves its physical index
// between the two bond indices, so this contraction is not a simple
// reshape).
void shift_center_right(MPSState& state, std::size_t p) {
    SiteTensor& a = state.site(p);
    const std::size_t old_chi_right = a.chi_right;
    const common::SvdResult svd_result = common::svd(a.data, a.chi_left * 2, a.chi_right);
    const std::size_t k = svd_result.singular_values.size();

    a.chi_right = k;
    a.data = svd_result.u;

    SiteTensor& b = state.site(p + 1);
    const std::size_t chi_right_b = b.chi_right;
    SiteTensor new_b{k, chi_right_b, std::vector<Complex>(k * 2 * chi_right_b, Complex{0.0, 0.0})};
    for (std::size_t lp = 0; lp < k; ++lp) {
        for (std::size_t s = 0; s < 2; ++s) {
            for (std::size_t r = 0; r < chi_right_b; ++r) {
                Complex sum{0.0, 0.0};
                for (std::size_t r0 = 0; r0 < old_chi_right; ++r0) {
                    const Complex sv =
                        svd_result.singular_values[lp] * svd_result.vt[lp * old_chi_right + r0];
                    sum += sv * b.at(r0, s, r);
                }
                new_b.at(lp, s, r) = sum;
            }
        }
    }
    b = std::move(new_b);
}

// Symmetric counterpart: moves the center from p to p-1. Site p's own
// data, viewed as a chi_left x (2*chi_right) row-major matrix, exactly
// matches its storage layout; Vt becomes the new (right-isometric)
// site p directly. U*S is absorbed into site p-1's right bond by
// explicit contraction, for the same reason as above.
void shift_center_left(MPSState& state, std::size_t p) {
    SiteTensor& a = state.site(p);
    const std::size_t old_chi_left = a.chi_left;
    const common::SvdResult svd_result = common::svd(a.data, a.chi_left, 2 * a.chi_right);
    const std::size_t k = svd_result.singular_values.size();

    a.chi_left = k;
    a.data = svd_result.vt;

    SiteTensor& c = state.site(p - 1);
    const std::size_t chi_left_c = c.chi_left;
    SiteTensor new_c{chi_left_c, k, std::vector<Complex>(chi_left_c * 2 * k, Complex{0.0, 0.0})};
    for (std::size_t l = 0; l < chi_left_c; ++l) {
        for (std::size_t s = 0; s < 2; ++s) {
            for (std::size_t lp = 0; lp < k; ++lp) {
                Complex sum{0.0, 0.0};
                for (std::size_t r0 = 0; r0 < old_chi_left; ++r0) {
                    const Complex us = svd_result.u[r0 * k + lp] * svd_result.singular_values[lp];
                    sum += c.at(l, s, r0) * us;
                }
                new_c.at(l, s, lp) = sum;
            }
        }
    }
    c = std::move(new_c);
}

}  // namespace

MPSState::MPSState(std::uint32_t num_qubits)
    : num_qubits_(num_qubits), orthogonality_center_(0), sites_(num_qubits) {
    common::ensure(num_qubits >= 1, "MPSState: num_qubits must be at least 1");
    for (SiteTensor& t : sites_) {
        t.chi_left = 1;
        t.chi_right = 1;
        t.data = {Complex{1.0, 0.0}, Complex{0.0, 0.0}};  // |0> in a trivial 1x2x1 tensor.
    }
    charge_sites();
}

void MPSState::charge_sites() {
    std::uint64_t bytes = common::buffer_bytes(sites_);
    for (const SiteTensor& t : sites_) {
        bytes += common::buffer_bytes(t.data);
    }
    charge_.set(bytes);
}

std::size_t MPSState::bond_dimension(std::size_t i) const {
    common::ensure(i + 1 < num_qubits_, "MPSState::bond_dimension: i + 1 out of range");
    return sites_[i].chi_right;
}

std::size_t MPSState::max_bond_dimension() const {
    if (num_qubits_ < 2) {
        return 1;
    }
    std::size_t result = 0;
    for (std::size_t i = 0; i + 1 < num_qubits_; ++i) {
        result = std::max(result, sites_[i].chi_right);
    }
    return result;
}

void MPSState::move_orthogonality_center_to(std::size_t target) {
    while (orthogonality_center_ < target) {
        shift_center_right(*this, orthogonality_center_);
        ++orthogonality_center_;
    }
    while (orthogonality_center_ > target) {
        shift_center_left(*this, orthogonality_center_);
        --orthogonality_center_;
    }
    charge_sites();
}

}  // namespace menos::engines::mps
