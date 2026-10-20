#include "engines/mps/expectation.h"

#include <array>
#include <vector>

#include "common/memory_ledger.h"
#include "common/pauli_row.h"

namespace menos::engines::mps {
namespace {

using Complex = std::complex<double>;

}  // namespace

std::complex<double> compute_expectation(const MPSState& state, const PauliString& pauli) {
    const std::uint32_t n = state.num_qubits();

    // L[l][l'] = the contraction of everything left of the current
    // site, bra bond l (conjugated) against ket bond l', with each
    // already-processed site's own Pauli matrix already inserted.
    // Starts as the trivial 1x1 identity (no sites processed yet).
    std::vector<Complex> l = {Complex{1.0, 0.0}};
    std::size_t chi = 1;
    common::MemoryLedger::Charge environment_charge;

    for (std::uint32_t i = 0; i < n; ++i) {
        const SiteTensor& t = state.site(i);
        const std::array<Complex, 4> p = common::pauli_matrix_2x2(
            common::get_bit(pauli.x_words, i), common::get_bit(pauli.z_words, i));

        std::vector<Complex> l_new(t.chi_right * t.chi_right, Complex{0.0, 0.0});
        environment_charge.set(common::buffer_bytes(l) + common::buffer_bytes(l_new));
        for (std::size_t r = 0; r < t.chi_right; ++r) {
            for (std::size_t rp = 0; rp < t.chi_right; ++rp) {
                Complex sum{0.0, 0.0};
                for (std::size_t bra_l = 0; bra_l < chi; ++bra_l) {
                    for (std::size_t ket_l = 0; ket_l < chi; ++ket_l) {
                        const Complex l_val = l[bra_l * chi + ket_l];
                        if (l_val == Complex{0.0, 0.0}) {
                            continue;
                        }
                        for (std::size_t s = 0; s < 2; ++s) {
                            for (std::size_t s2 = 0; s2 < 2; ++s2) {
                                sum += std::conj(t.at(bra_l, s, r)) * l_val * p[s * 2 + s2] *
                                       t.at(ket_l, s2, rp);
                            }
                        }
                    }
                }
                l_new[r * t.chi_right + rp] = sum;
            }
        }
        l = std::move(l_new);
        environment_charge.set(common::buffer_bytes(l));
        chi = t.chi_right;
    }

    return l[0];  // chi == 1 here (state's own right boundary).
}

}  // namespace menos::engines::mps
