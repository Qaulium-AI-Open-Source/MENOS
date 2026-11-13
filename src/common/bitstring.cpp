#include "menos/bitstring.h"

#include <iomanip>
#include <sstream>

#include "common/error.h"

namespace menos {

namespace {
constexpr std::uint32_t kBitsPerWord = 64;

[[nodiscard]] std::size_t word_count(std::uint32_t num_bits) {
    return (static_cast<std::size_t>(num_bits) + kBitsPerWord - 1) / kBitsPerWord;
}
}  // namespace

BitString::BitString(std::uint32_t num_bits)
    : num_bits_(num_bits), words_(word_count(num_bits), std::uint64_t{0}) {}

std::uint32_t BitString::num_bits() const {
    return num_bits_;
}

bool BitString::test(std::uint32_t bit) const {
    common::ensure(bit < num_bits_, "BitString::test: bit index out of range");
    const std::size_t word = bit / kBitsPerWord;
    const std::uint32_t offset = bit % kBitsPerWord;
    return ((words_[word] >> offset) & std::uint64_t{1}) != 0;
}

void BitString::set(std::uint32_t bit, bool value) {
    common::ensure(bit < num_bits_, "BitString::set: bit index out of range");
    const std::size_t word = bit / kBitsPerWord;
    const std::uint32_t offset = bit % kBitsPerWord;
    if (value) {
        words_[word] |= (std::uint64_t{1} << offset);
    } else {
        words_[word] &= ~(std::uint64_t{1} << offset);
    }
}

BitString& BitString::operator|=(const BitString& other) {
    common::ensure(other.num_bits_ == num_bits_,
                   "BitString::operator|=: operands must have the same width");
    for (std::size_t i = 0; i < words_.size(); ++i) {
        words_[i] |= other.words_[i];
    }
    return *this;
}

BitString& BitString::operator^=(const BitString& other) {
    common::ensure(other.num_bits_ == num_bits_,
                   "BitString::operator^=: operands must have the same width");
    for (std::size_t i = 0; i < words_.size(); ++i) {
        words_[i] ^= other.words_[i];
    }
    return *this;
}

bool operator==(const BitString& lhs, const BitString& rhs) {
    return lhs.num_bits_ == rhs.num_bits_ && lhs.words_ == rhs.words_;
}

std::strong_ordering operator<=>(const BitString& lhs, const BitString& rhs) {
    common::ensure(lhs.num_bits_ == rhs.num_bits_,
                   "BitString::operator<=>: operands must have the same width");
    for (std::size_t i = lhs.words_.size(); i-- > 0;) {
        if (lhs.words_[i] != rhs.words_[i]) {
            return lhs.words_[i] <=> rhs.words_[i];
        }
    }
    return std::strong_ordering::equal;
}

std::string BitString::to_hex_string() const {
    std::size_t highest = 0;
    bool any_nonzero = false;
    for (std::size_t i = words_.size(); i-- > 0;) {
        if (words_[i] != 0) {
            highest = i;
            any_nonzero = true;
            break;
        }
    }
    if (!any_nonzero) {
        return "0x0";
    }

    std::ostringstream oss;
    oss << "0x" << std::hex << words_[highest];
    for (std::size_t i = highest; i-- > 0;) {
        oss << std::setw(kBitsPerWord / 4) << std::setfill('0') << std::hex << words_[i];
    }
    return oss.str();
}

}  // namespace menos
