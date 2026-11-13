#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <vector>

namespace menos::common {

// A single word-packed bit vector's bit q, q/64 into words, q%64 within
// the word -- the storage convention every bit-packed type in this
// codebase uses (BitString, Tableau's rows). Shared here because two
// independent consumers now read/write these bits directly: PauliRow's
// own apply_*() below, and the stabilizer tableau's Gauss-Jordan
// elimination (src/engines/stabilizer/tableau.cpp), which inspects and
// combines PauliRow bits directly rather than through apply_*().
//
// Inline and branch-free (docs/adr/0028 section 22): the stabilizer and
// Pauli propagation engines call them for every row of every gate, and a
// branch on a random bit costs a misprediction half the time, which made
// the per-update cost depend on the rows' contents.
[[nodiscard]] inline bool get_bit(const std::vector<std::uint64_t>& words, std::uint32_t q) {
    return ((words[q / 64] >> (q % 64)) & std::uint64_t{1}) != 0;
}
inline void set_bit(std::vector<std::uint64_t>& words, std::uint32_t q, bool value) {
    const std::uint32_t shift = q % 64;
    std::uint64_t& word = words[q / 64];
    word = (word & ~(std::uint64_t{1} << shift)) | (static_cast<std::uint64_t>(value) << shift);
}
inline void xor_bit(std::vector<std::uint64_t>& words, std::uint32_t q, bool value) {
    words[q / 64] ^= static_cast<std::uint64_t>(value) << (q % 64);
}

// One Pauli string's bit-packed representation: per qubit j, (x,z) bits
// at position j of the x/z word vectors give I/X/Z/Y as (0,0)/(1,0)/
// (0,1)/(1,1), with Y = i*X*Z exactly (docs/adr/0006's convention,
// verified there against src/ir/gate_table.cpp's matrix_y). r is the
// overall sign: false for +1, true for -1. Shared by two independent
// consumers (docs/BUILD_PLAN.md Phase 5, "abstractions follow the
// second implementation"): the stabilizer tableau, which loops this
// over its n generator rows, and the Pauli propagation engine, which
// applies it to a list of weighted observable terms.
struct PauliRow {
    std::vector<std::uint64_t> x;
    std::vector<std::uint64_t> z;
    bool r;
};

// Each apply_*() implements the per-gate conjugation rule derived and
// verified in docs/adr/0006 by direct matrix computation, touching only
// the given qubit(s)' bits -- O(1) words, independent of how many
// qubits the row spans.
void apply_x(PauliRow& row, std::uint32_t a);
void apply_y(PauliRow& row, std::uint32_t a);
void apply_z(PauliRow& row, std::uint32_t a);
void apply_h(PauliRow& row, std::uint32_t a);
void apply_s(PauliRow& row, std::uint32_t a);
void apply_sdg(PauliRow& row, std::uint32_t a);
void apply_sx(PauliRow& row, std::uint32_t a);
void apply_sxdg(PauliRow& row, std::uint32_t a);
// Precondition: control != target (checked via common::ensure ->
// InternalError).
void apply_cx(PauliRow& row, std::uint32_t control, std::uint32_t target);
void apply_cz(PauliRow& row, std::uint32_t a, std::uint32_t b);
void apply_swap(PauliRow& row, std::uint32_t a, std::uint32_t b);

// Power of i in sigma1*sigma2 = i^g * sigma(x1^x2, z1^z2), derived and
// tabulated in docs/adr/0006 by direct 2x2 matrix multiplication for
// every (sigma1, sigma2) pair (sigma in {I, X, Y, Z} via (x,z) bits).
// Exposed here (not file-local) because both Tableau::rowsum and the
// stabilizer expectation reduction (docs/adr/0007) need it.
[[nodiscard]] int pauli_phase_exponent(bool x1, bool z1, bool x2, bool z2);

// The sum over every qubit of pauli_phase_exponent(a's bits, b's bits),
// mod 4: the power of i in the product a*b, signs excluded. Computed a
// word at a time (docs/adr/0028 section 22): per word, the qubits where
// the factors anticommute contribute 1 or 3, by the cyclic order
// X -> Y -> Z, and the two masks are counted. Exact integer arithmetic, so
// it equals the per-qubit sum, which tests/common/pauli_row_test.cpp
// keeps as the oracle. Precondition: a and b have the same word count, and
// bits past the qubit count are zero in both.
[[nodiscard]] int pauli_product_phase_exponent(const PauliRow& a, const PauliRow& b);

// The single-qubit Pauli matrix for bits (x, z), row-major 2x2, matching
// PauliString's own documented convention (I/X/Z/Y for (0,0)/(1,0)/
// (0,1)/(1,1), Y = i*X*Z exactly). Used by the MPS engine's own
// expectation contraction (src/engines/mps/expectation.cpp).
[[nodiscard]] std::array<std::complex<double>, 4> pauli_matrix_2x2(bool x, bool z);

}  // namespace menos::common
