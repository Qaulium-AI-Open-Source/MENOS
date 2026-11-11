#include "common/pauli_row.h"

#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace menos::common {
namespace {

using Complex = std::complex<double>;

PauliRow make_row(bool x0, bool z0) {
    PauliRow row{std::vector<std::uint64_t>(1, 0), std::vector<std::uint64_t>(1, 0), false};
    set_bit(row.x, 0, x0);
    set_bit(row.z, 0, z0);
    return row;
}

TEST(PauliRowTest, ApplyXFlipsSignOnlyWhenRowHasZComponent) {
    PauliRow z_row = make_row(false, true);  // Z
    apply_x(z_row, 0);
    EXPECT_TRUE(z_row.r);

    PauliRow x_row = make_row(true, false);  // X
    apply_x(x_row, 0);
    EXPECT_FALSE(x_row.r);
}

TEST(PauliRowTest, ApplyHSwapsXAndZBits) {
    PauliRow row = make_row(true, false);  // X
    apply_h(row, 0);
    EXPECT_FALSE(get_bit(row.x, 0));
    EXPECT_TRUE(get_bit(row.z, 0));
    EXPECT_FALSE(row.r);
}

// Conjugating a Z row by the circuit H,Z,H gives (HZH)*Z*(HZH)dagger =
// X*Z*X = -Z exactly (HZH=X as an operator, matches
// tests/engines/stabilizer/tableau_test.cpp's HZHEqualsXOnDeterministicOutcome,
// which only checks the resulting measurement outcome; here the check is
// the stronger claim that conjugating by X directly gives the identical
// row, not just the same outcome).
TEST(PauliRowTest, HZHConjugationMatchesDirectXConjugation) {
    PauliRow via_hzh = make_row(false, true);  // Z
    apply_h(via_hzh, 0);
    apply_z(via_hzh, 0);
    apply_h(via_hzh, 0);

    PauliRow via_x = make_row(false, true);  // Z
    apply_x(via_x, 0);

    EXPECT_EQ(via_hzh.x, via_x.x);
    EXPECT_EQ(via_hzh.z, via_x.z);
    EXPECT_EQ(via_hzh.r, via_x.r);
    // Both conjugate Z into -Z: x=0 (unchanged), z=1 (unchanged), sign
    // flipped to true (-1).
    EXPECT_FALSE(get_bit(via_hzh.x, 0));
    EXPECT_TRUE(get_bit(via_hzh.z, 0));
    EXPECT_TRUE(via_hzh.r);
}

TEST(PauliRowTest, SxThenSxdgIsIdentity) {
    PauliRow row = make_row(false, true);  // Z
    const auto original_x = row.x;
    const auto original_z = row.z;
    const bool original_r = row.r;
    apply_sx(row, 0);
    apply_sxdg(row, 0);
    EXPECT_EQ(row.x, original_x);
    EXPECT_EQ(row.z, original_z);
    EXPECT_EQ(row.r, original_r);
}

TEST(PauliRowTest, CxOnXaZbGivesMinusYaYaWithFlip) {
    // (x_a=1,z_a=0,x_b=0,z_b=1) = X_a (x) Z_b -> CX -a,b-> -Y_a Y_b,
    // the tightest hand-derived case in docs/adr/0006.
    PauliRow row{std::vector<std::uint64_t>(1, 0), std::vector<std::uint64_t>(1, 0), false};
    set_bit(row.x, 0, true);
    set_bit(row.z, 1, true);
    apply_cx(row, 0, 1);
    EXPECT_TRUE(get_bit(row.x, 0));
    EXPECT_TRUE(get_bit(row.z, 0));
    EXPECT_TRUE(get_bit(row.x, 1));
    EXPECT_TRUE(get_bit(row.z, 1));
    EXPECT_TRUE(row.r);
}

TEST(PauliRowTest, PauliPhaseExponentMatchesTheDocumentedTable) {
    // g(X,Z)=3, g(Z,X)=1, g(X,Y)=1, g(Y,X)=3, g(Z,Y)=3, g(Y,Z)=1
    // (docs/adr/0006's table, sigma encoded as (x,z)).
    EXPECT_EQ(pauli_phase_exponent(true, false, false, true), 3);  // X,Z
    EXPECT_EQ(pauli_phase_exponent(false, true, true, false), 1);  // Z,X
    EXPECT_EQ(pauli_phase_exponent(true, false, true, true), 1);   // X,Y
    EXPECT_EQ(pauli_phase_exponent(true, true, true, false), 3);   // Y,X
    EXPECT_EQ(pauli_phase_exponent(false, true, true, true), 3);   // Z,Y
    EXPECT_EQ(pauli_phase_exponent(true, true, false, true), 1);   // Y,Z
    EXPECT_EQ(pauli_phase_exponent(false, false, true, true), 0);  // I,Y
    EXPECT_EQ(pauli_phase_exponent(true, true, true, true), 0);    // Y,Y
}

// The per-qubit sum the stabilizer's rowsum computed until Phase 24: the
// oracle for the word-parallel pauli_product_phase_exponent.
int per_qubit_phase(const PauliRow& a, const PauliRow& b, std::uint32_t n) {
    int total = 0;
    for (std::uint32_t q = 0; q < n; ++q) {
        total += pauli_phase_exponent(
            get_bit(a.x, q), get_bit(a.z, q), get_bit(b.x, q), get_bit(b.z, q));
    }
    return total % 4;
}

PauliRow empty_row(std::uint32_t n) {
    const std::size_t words = (static_cast<std::size_t>(n) + 63) / 64;
    return PauliRow{
        std::vector<std::uint64_t>(words, 0), std::vector<std::uint64_t>(words, 0), false};
}

// docs/adr/0028 section 22: every ordered pair of single-qubit Paulis, at
// the first, a middle and the last bit of a word and past a word boundary.
TEST(PauliRowTest, ProductPhaseExponentMatchesThePerQubitTableOnEveryPair) {
    for (const std::uint32_t q : {0U, 17U, 63U, 64U, 100U}) {
        for (int p1 = 0; p1 < 4; ++p1) {
            for (int p2 = 0; p2 < 4; ++p2) {
                PauliRow a = empty_row(101);
                PauliRow b = empty_row(101);
                set_bit(a.x, q, (p1 & 1) != 0);
                set_bit(a.z, q, (p1 & 2) != 0);
                set_bit(b.x, q, (p2 & 1) != 0);
                set_bit(b.z, q, (p2 & 2) != 0);
                EXPECT_EQ(pauli_product_phase_exponent(a, b), per_qubit_phase(a, b, 101))
                    << "qubit " << q << " pair " << p1 << "," << p2;
            }
        }
    }
}

TEST(PauliRowTest, ProductPhaseExponentMatchesThePerQubitSumOnRandomRows) {
    std::uint64_t state = 0x243F6A8885A308D3ULL;
    const auto next = [&state] {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (const std::uint32_t n : {1U, 63U, 64U, 65U, 200U}) {
        for (int trial = 0; trial < 200; ++trial) {
            PauliRow a = empty_row(n);
            PauliRow b = empty_row(n);
            for (std::uint32_t q = 0; q < n; ++q) {
                const std::uint64_t r = next();
                set_bit(a.x, q, (r & 1) != 0);
                set_bit(a.z, q, (r & 2) != 0);
                set_bit(b.x, q, (r & 4) != 0);
                set_bit(b.z, q, (r & 8) != 0);
            }
            EXPECT_EQ(pauli_product_phase_exponent(a, b), per_qubit_phase(a, b, n)) << "n " << n;
        }
    }
}

// docs/BUILD_PLAN.md Phase 8: pauli_matrix_2x2 is now a shared function
// (promoted from src/engines/mps/expectation.cpp's own anonymous
// namespace once a second consumer, the tensor-network engine, needed
// it) -- direct coverage here rather than relying only on MPS's own
// expectation tests exercising it indirectly.
TEST(PauliRowTest, PauliMatrix2x2MatchesIXZYForEachBitCombination) {
    EXPECT_EQ(pauli_matrix_2x2(false, false),
              (std::array<Complex, 4>{Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, Complex{1, 0}}));
    EXPECT_EQ(pauli_matrix_2x2(true, false),
              (std::array<Complex, 4>{Complex{0, 0}, Complex{1, 0}, Complex{1, 0}, Complex{0, 0}}));
    EXPECT_EQ(
        pauli_matrix_2x2(false, true),
        (std::array<Complex, 4>{Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, Complex{-1, 0}}));
    // Y = i*X*Z exactly (CLAUDE.md's Pauli phase convention).
    EXPECT_EQ(
        pauli_matrix_2x2(true, true),
        (std::array<Complex, 4>{Complex{0, 0}, Complex{0, -1}, Complex{0, 1}, Complex{0, 0}}));
}

}  // namespace
}  // namespace menos::common
