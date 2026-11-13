#include "menos/bitstring.h"

#include <gtest/gtest.h>

#include "common/error.h"

namespace menos {
namespace {

TEST(BitStringTest, StartsAllZero) {
    const BitString bs(8);
    EXPECT_EQ(bs.num_bits(), 8U);
    for (std::uint32_t i = 0; i < 8; ++i) {
        EXPECT_FALSE(bs.test(i)) << "bit " << i;
    }
}

TEST(BitStringTest, SetAndTestRoundTrip) {
    BitString bs(10);
    bs.set(0, true);
    bs.set(7, true);
    bs.set(9, true);
    EXPECT_TRUE(bs.test(0));
    EXPECT_FALSE(bs.test(1));
    EXPECT_TRUE(bs.test(7));
    EXPECT_FALSE(bs.test(8));
    EXPECT_TRUE(bs.test(9));

    bs.set(7, false);
    EXPECT_FALSE(bs.test(7));
}

TEST(BitStringTest, SetAcrossWordBoundary) {
    // 100 bits spans two 64-bit words; bit 64 is the first bit of word 1.
    BitString bs(100);
    bs.set(63, true);
    bs.set(64, true);
    bs.set(99, true);
    EXPECT_TRUE(bs.test(63));
    EXPECT_TRUE(bs.test(64));
    EXPECT_TRUE(bs.test(99));
    EXPECT_FALSE(bs.test(65));
}

TEST(BitStringTest, OutOfRangeTestThrowsInternalError) {
    const BitString bs(4);
    EXPECT_THROW(static_cast<void>(bs.test(4)), common::InternalError);
}

TEST(BitStringTest, OutOfRangeSetThrowsInternalError) {
    BitString bs(4);
    EXPECT_THROW(bs.set(4, true), common::InternalError);
}

TEST(BitStringTest, OrCombinesDisjointBits) {
    BitString a(8);
    a.set(0, true);
    BitString b(8);
    b.set(7, true);
    a |= b;
    EXPECT_TRUE(a.test(0));
    EXPECT_TRUE(a.test(7));
    for (std::uint32_t i = 1; i < 7; ++i) {
        EXPECT_FALSE(a.test(i));
    }
}

TEST(BitStringTest, XorFlipsBitsInBothDirections) {
    BitString a(70);
    a.set(0, true);
    a.set(69, true);
    BitString b(70);
    b.set(0, true);   // 1 ^ 1 = 0
    b.set(64, true);  // 0 ^ 1 = 1, across the word boundary
    a ^= b;
    EXPECT_FALSE(a.test(0));
    EXPECT_TRUE(a.test(64));
    EXPECT_TRUE(a.test(69));
    a ^= b;  // an involution
    EXPECT_TRUE(a.test(0));
    EXPECT_FALSE(a.test(64));
    EXPECT_TRUE(a.test(69));
}

TEST(BitStringTest, XorMismatchedWidthThrowsInternalError) {
    BitString a(8);
    BitString b(9);
    EXPECT_THROW(a ^= b, common::InternalError);
}

TEST(BitStringTest, OrMismatchedWidthThrowsInternalError) {
    BitString a(8);
    BitString b(9);
    EXPECT_THROW(a |= b, common::InternalError);
}

TEST(BitStringTest, EqualityConsidersWidthAndBits) {
    BitString a(8);
    BitString b(8);
    EXPECT_EQ(a, b);
    a.set(3, true);
    EXPECT_NE(a, b);
    b.set(3, true);
    EXPECT_EQ(a, b);
}

TEST(BitStringTest, OrderingIsNumericAcrossWordBoundary) {
    BitString low(100);
    low.set(0, true);  // value 1
    BitString high(100);
    high.set(99, true);  // value 2^99, in the second word
    EXPECT_LT(low, high);
    EXPECT_GT(high, low);

    BitString equal_to_low(100);
    equal_to_low.set(0, true);
    EXPECT_EQ(std::strong_ordering::equal, low <=> equal_to_low);
}

TEST(BitStringTest, ComparisonMismatchedWidthThrowsInternalError) {
    const BitString a(8);
    const BitString b(9);
    EXPECT_THROW(static_cast<void>(a <=> b), common::InternalError);
}

TEST(BitStringTest, ToHexStringZero) {
    const BitString bs(8);
    EXPECT_EQ(bs.to_hex_string(), "0x0");
}

TEST(BitStringTest, ToHexStringSmallValue) {
    BitString bs(8);
    bs.set(0, true);
    bs.set(1, true);
    bs.set(3, true);  // 0b1011 = 0xb
    EXPECT_EQ(bs.to_hex_string(), "0xb");
}

TEST(BitStringTest, ToHexStringNoLeadingZeroPadding) {
    // 1 bit set at position 8: value 0x100, not "0x0100".
    BitString bs(64);
    bs.set(8, true);
    EXPECT_EQ(bs.to_hex_string(), "0x100");
}

TEST(BitStringTest, ToHexStringSpansWordBoundaryWithPadding) {
    // Bit 64 set (second word, value 1) and bit 0 set (first word, value
    // 1): represents 2^64 + 1 = 0x1_0000000000000001. The low word must
    // be zero-padded to 16 hex digits since it's not the highest word.
    BitString bs(65);
    bs.set(0, true);
    bs.set(64, true);
    EXPECT_EQ(bs.to_hex_string(), "0x10000000000000001");
}

TEST(BitStringTest, WidthBoundaries63_64_65_100) {
    for (const std::uint32_t width : {63U, 64U, 65U, 100U}) {
        BitString bs(width);
        EXPECT_EQ(bs.num_bits(), width);
        bs.set(width - 1, true);
        EXPECT_TRUE(bs.test(width - 1));
        EXPECT_FALSE(bs.test(0));
        EXPECT_THROW(static_cast<void>(bs.test(width)), common::InternalError);
    }
}

TEST(BitStringTest, WidthOneAndSeven) {
    for (const std::uint32_t width : {1U, 7U}) {
        BitString bs(width);
        bs.set(0, true);
        EXPECT_TRUE(bs.test(0));
        EXPECT_EQ(bs.to_hex_string(), "0x1");
    }
}

}  // namespace
}  // namespace menos
