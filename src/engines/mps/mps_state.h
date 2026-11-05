#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/memory_ledger.h"

namespace menos::engines::mps {

// One site's tensor, dims (chi_left, 2, chi_right), row-major:
// data[(l * 2 + s) * chi_right + r] is the amplitude contribution for
// left-bond index l, physical index s (0 or 1), right-bond index r.
struct SiteTensor {
    std::size_t chi_left = 1;
    std::size_t chi_right = 1;
    std::vector<std::complex<double>> data;  // size chi_left * 2 * chi_right

    [[nodiscard]] std::complex<double>& at(std::size_t l, std::size_t s, std::size_t r) {
        return data[(l * 2 + s) * chi_right + r];
    }
    [[nodiscard]] std::complex<double> at(std::size_t l, std::size_t s, std::size_t r) const {
        return data[(l * 2 + s) * chi_right + r];
    }
};

// A canonical (orthogonality-center) tensor-train state for num_qubits
// qubits. The qubit-to-site mapping is fixed for the object's whole
// lifetime: site i is always qubit i (docs/BUILD_PLAN.md Phase 6:
// reordering qubits to reduce cut size is out of scope). Starts at
// |0...0>: every site trivial (chi_left == chi_right == 1, data = [1]),
// orthogonality_center at site 0 (the whole state is already both
// left- and right-canonical everywhere when every bond is trivial).
//
// MPS gauge (CLAUDE.md domain invariant): SVD gauge freedom means two
// MPSState objects representing the same physical state are not
// required to hold bit-identical tensors, even from the same LAPACK
// implementation across platforms. Nothing in this class or its
// callers may compare raw tensor values for correctness; only physical
// outputs (sampled counts, expectation values, computed from
// gauge-invariant contractions) are required to be deterministic.
class MPSState {
public:
    explicit MPSState(std::uint32_t num_qubits);

    [[nodiscard]] std::uint32_t num_qubits() const {
        return num_qubits_;
    }
    [[nodiscard]] std::size_t orthogonality_center() const {
        return orthogonality_center_;
    }
    [[nodiscard]] const SiteTensor& site(std::size_t i) const {
        return sites_[i];
    }
    [[nodiscard]] SiteTensor& site(std::size_t i) {
        return sites_[i];
    }

    // The bond dimension between site i and site i+1 (site(i).chi_right,
    // maintained equal to site(i + 1).chi_left by every mutator).
    // Precondition: i + 1 < num_qubits().
    [[nodiscard]] std::size_t bond_dimension(std::size_t i) const;

    // The largest bond dimension anywhere in the chain (1 for a
    // num_qubits() == 1 state, which has no bonds).
    [[nodiscard]] std::size_t max_bond_dimension() const;

    // Moves the orthogonality center to target via a chain of exact
    // (untruncated) thin-SVD-based shifts, one site at a time -- always
    // lossless (every singular value is kept, none discarded), since
    // this only repositions where the state's weight is carried, it
    // never applies a gate or approximates. A no-op if already at
    // target.
    void move_orthogonality_center_to(std::size_t target);

    // Low-level bookkeeping only, for engines/mps/apply_gate.cpp: after
    // it replaces a contiguous block's own site tensors directly via
    // site(i) (its own multi-site SVD update, which computes a new
    // orthogonality-center position as part of that update, unlike
    // move_orthogonality_center_to's own step-by-step shifts), it
    // updates the bookkeeping here. Does not itself touch any tensor;
    // the caller is responsible for the tensors already being
    // consistent with center being at site.
    void set_orthogonality_center_unchecked(std::size_t site) {
        orthogonality_center_ = site;
    }

    // Charges the site tensors' current bytes to the memory ledger
    // (docs/adr/0025 section 3.2), replacing the previous figure. Called by
    // the constructor and move_orthogonality_center_to, and by the engine
    // after each gate, since apply_gate replaces sites through site(i).
    void charge_sites();

private:
    std::uint32_t num_qubits_;
    std::size_t orthogonality_center_;
    std::vector<SiteTensor> sites_;
    common::MemoryLedger::Charge charge_;
};

}  // namespace menos::engines::mps
