#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "common/pauli_row.h"
#include "menos/observable.h"

namespace menos::engines::stabilizer {

// Aaronson-Gottesman-style stabilizer tableau, bit-packed, storing only
// the `n` stabilizer generators (no destabilizers -- see docs/adr/0006
// for why MENOS's terminal-measurement-only semantics make them
// unnecessary). Initial state |0...0>: row i is Z_i (x=0, z=e_i, r=0).
// Rows are common::PauliRow (docs/BUILD_PLAN.md Phase 5: shared with
// the Pauli propagation engine, "abstractions follow the second
// implementation").
//
// Every apply_*() delegates to common:: (docs/adr/0006's ten gate
// rules), looped over every row -- O(1) words per row, never
// per-amplitude.
class Tableau {
public:
    explicit Tableau(std::uint32_t num_qubits);

    void apply_x(std::uint32_t a);
    void apply_y(std::uint32_t a);
    void apply_z(std::uint32_t a);
    void apply_h(std::uint32_t a);
    void apply_s(std::uint32_t a);
    void apply_sdg(std::uint32_t a);
    void apply_sx(std::uint32_t a);
    void apply_sxdg(std::uint32_t a);
    // Precondition: control != target (checked via common::ensure ->
    // InternalError -- validated once at the KIR boundary, not
    // re-checked per call site).
    void apply_cx(std::uint32_t control, std::uint32_t target);
    void apply_cz(std::uint32_t a, std::uint32_t b);
    void apply_swap(std::uint32_t a, std::uint32_t b);

    [[nodiscard]] std::uint32_t num_qubits() const;

    // The affine subspace of full n-qubit Z-basis measurement outcomes
    // (docs/adr/0006's Gaussian-elimination reduction): `base` (n bits)
    // is one valid outcome, `generators` (r independent n-bit vectors,
    // r = affine dimension) span the subspace's linear part. Every
    // outcome is `base` XOR (XOR of some subset of `generators`), each
    // subset giving a distinct, equally likely outcome. Does not mutate
    // this tableau (reduces a copy).
    struct AffineSubspace {
        std::vector<bool> base;
        std::vector<std::vector<bool>> generators;
    };
    [[nodiscard]] AffineSubspace reduce() const;

    // The exact expectation of a Pauli string on this (fully Clifford)
    // region's state, docs/adr/0007's GF(2)-elimination reduction:
    // nullopt if query anticommutes with some stabilizer generator
    // (expectation exactly 0); otherwise the sign (false: +1, true: -1).
    // Precondition: query.num_qubits() == num_qubits().
    [[nodiscard]] std::optional<bool> expectation_sign(const PauliString& query) const;

    // Upper bounds on the heap bytes these allocate for an n-qubit tableau
    // (Phase 21, docs/adr/0025 section 3.4), from the types they use:
    // the tableau's own rows; reduce()'s working copies and its returned
    // subspace, at the end of reduce() when all are live; the returned
    // subspace alone; and expectation_sign()'s working state for one term.
    [[nodiscard]] static std::uint64_t rows_bytes(std::uint32_t num_qubits);
    [[nodiscard]] static std::uint64_t reduce_peak_bytes(std::uint32_t num_qubits);
    [[nodiscard]] static std::uint64_t subspace_bytes(std::uint32_t num_qubits);
    [[nodiscard]] static std::uint64_t expectation_peak_bytes(std::uint32_t num_qubits);

    // The bytes this tableau's rows asked for, as the memory ledger charges
    // them (docs/adr/0025 section 3.2).
    [[nodiscard]] std::uint64_t held_bytes() const;

private:
    [[nodiscard]] std::size_t words_per_row() const;

    std::uint32_t num_qubits_;
    std::vector<common::PauliRow> rows_;  // rows_[i]: stabilizer generator i, n rows total
};

}  // namespace menos::engines::stabilizer
