#pragma once

#include <compare>
#include <cstdint>
#include <string>
#include <vector>

namespace menos {

// An arbitrary-width, unsigned classical bit string: Phase 3's production
// count-key representation, replacing the Phase 1 uint64_t key and its
// 64-classical-bit ceiling (docs/BUILD_PLAN.md Phase 3, 3.4/3.8).
//
// Logical bit index c maps to bit c of the represented value, matching
// CLAUDE.md's classical-bit domain invariant (bit c of a counts key is
// classical bit c). Storage is little-endian by 64-bit word: word 0 holds
// bits [0, 64), word 1 holds bits [64, 128), and so on.
class BitString {
public:
    // All bits initialized to 0.
    explicit BitString(std::uint32_t num_bits);

    [[nodiscard]] std::uint32_t num_bits() const;

    // Precondition: bit < num_bits(). Checked via common::ensure ->
    // InternalError.
    [[nodiscard]] bool test(std::uint32_t bit) const;
    void set(std::uint32_t bit, bool value);

    // Bitwise OR, in place. Precondition: other.num_bits() == num_bits().
    // Checked via common::ensure -> InternalError. Used to combine
    // per-region factorized-counts contributions, which occupy disjoint
    // bit ranges by construction (docs/BUILD_PLAN.md 3.4).
    BitString& operator|=(const BitString& other);

    // Bitwise XOR, in place. Precondition: other.num_bits() == num_bits().
    // Checked via common::ensure -> InternalError. Used by the stabilizer
    // engine's counts sampler to flip a generator into the base outcome
    // (docs/adr/0006: every outcome is base XOR a subset of generators; an OR
    // here was the Phase 24 sampler bug, docs/adr/0028 section 18).
    BitString& operator^=(const BitString& other);

    friend bool operator==(const BitString& lhs, const BitString& rhs);

    // Numeric lexicographic-by-most-significant-word: compares the
    // highest-index word first, so std::map<BitString, ...> iterates in
    // ascending numeric order, matching Phase 1's std::map<uint64_t, ...>
    // iteration order. Precondition: lhs.num_bits() == rhs.num_bits()
    // (checked via common::ensure -> InternalError) -- comparing
    // BitStrings of different widths is not a defined operation.
    friend std::strong_ordering operator<=>(const BitString& lhs, const BitString& rhs);

    // Lowercase "0x"-prefixed hexadecimal, byte-identical to Python's
    // built-in hex() on the same represented integer (no leading zero
    // padding beyond the single digit "0" for the zero value).
    [[nodiscard]] std::string to_hex_string() const;

private:
    std::uint32_t num_bits_;
    std::vector<std::uint64_t> words_;  // words_[0] = least-significant 64 bits
};

}  // namespace menos
