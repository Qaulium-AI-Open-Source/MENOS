#include "common/pauli_row.h"

#include <bit>
#include <cstddef>

#include "common/error.h"

namespace menos::common {

// Every rule below flips the sign with a branch-free XOR (docs/adr/0028
// section 22): `flip` is the rule's condition on the row's bits.
namespace {
void flip_sign(PauliRow& row, bool flip) {
    row.r = row.r != flip;
}
}  // namespace

void apply_x(PauliRow& row, std::uint32_t a) {
    flip_sign(row, get_bit(row.z, a));
}

void apply_z(PauliRow& row, std::uint32_t a) {
    flip_sign(row, get_bit(row.x, a));
}

void apply_y(PauliRow& row, std::uint32_t a) {
    flip_sign(row, get_bit(row.x, a) != get_bit(row.z, a));
}

void apply_h(PauliRow& row, std::uint32_t a) {
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    flip_sign(row, xa & za);
    set_bit(row.x, a, za);
    set_bit(row.z, a, xa);
}

void apply_s(PauliRow& row, std::uint32_t a) {
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    flip_sign(row, xa & za);
    xor_bit(row.z, a, xa);
}

void apply_sdg(PauliRow& row, std::uint32_t a) {
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    flip_sign(row, xa & !za);
    xor_bit(row.z, a, xa);
}

void apply_sx(PauliRow& row, std::uint32_t a) {
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    flip_sign(row, !xa & za);
    xor_bit(row.x, a, za);
}

void apply_sxdg(PauliRow& row, std::uint32_t a) {
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    flip_sign(row, xa & za);
    xor_bit(row.x, a, za);
}

void apply_cx(PauliRow& row, std::uint32_t control, std::uint32_t target) {
    ensure(control != target, "apply_cx: control and target must differ");
    const bool xa = get_bit(row.x, control);
    const bool za = get_bit(row.z, control);
    const bool xb = get_bit(row.x, target);
    const bool zb = get_bit(row.z, target);
    flip_sign(row, xa & zb & (xb == za));
    xor_bit(row.x, target, xa);
    xor_bit(row.z, control, zb);
}

void apply_cz(PauliRow& row, std::uint32_t a, std::uint32_t b) {
    ensure(a != b, "apply_cz: the two qubits must differ");
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    const bool xb = get_bit(row.x, b);
    const bool zb = get_bit(row.z, b);
    flip_sign(row, xa & xb & (za != zb));
    xor_bit(row.z, a, xb);
    xor_bit(row.z, b, xa);
}

void apply_swap(PauliRow& row, std::uint32_t a, std::uint32_t b) {
    ensure(a != b, "apply_swap: the two qubits must differ");
    const bool xa = get_bit(row.x, a);
    const bool za = get_bit(row.z, a);
    const bool xb = get_bit(row.x, b);
    const bool zb = get_bit(row.z, b);
    set_bit(row.x, a, xb);
    set_bit(row.x, b, xa);
    set_bit(row.z, a, zb);
    set_bit(row.z, b, za);
}

// sigma1*sigma2 = i^g * sigma(x1^x2, z1^z2), derived and tabulated in
// docs/adr/0006 by direct 2x2 matrix multiplication for every
// (sigma1, sigma2) pair.
int pauli_phase_exponent(bool x1, bool z1, bool x2, bool z2) {
    if ((!x1 && !z1) || (!x2 && !z2)) {
        return 0;  // either operand is I
    }
    if (x1 == x2 && z1 == z2) {
        return 0;  // same nontrivial type: X*X, Y*Y, Z*Z
    }
    if (x1 && !z1) {  // sigma1 = X, sigma2 in {Z, Y}
        return x2 ? 1 : 3;
    }
    if (!x1 && z1) {  // sigma1 = Z, sigma2 in {X, Y}
        return z2 ? 3 : 1;
    }
    // sigma1 = Y, sigma2 in {X, Z}
    return x2 ? 3 : 1;
}

int pauli_product_phase_exponent(const PauliRow& a, const PauliRow& b) {
    ensure(a.x.size() == b.x.size() && a.z.size() == b.z.size() && a.x.size() == a.z.size(),
           "pauli_product_phase_exponent: rows of different widths");
    // Per qubit, pauli_phase_exponent is 0 unless the factors anticommute,
    // then 1 for the cyclic pairs X*Y, Y*Z, Z*X and 3 for the others.
    std::uint64_t plus = 0;
    std::uint64_t minus = 0;
    for (std::size_t w = 0; w < a.x.size(); ++w) {
        const std::uint64_t x1 = a.x[w];
        const std::uint64_t z1 = a.z[w];
        const std::uint64_t x2 = b.x[w];
        const std::uint64_t z2 = b.z[w];
        const std::uint64_t anti = (x1 & z2) ^ (z1 & x2);
        const std::uint64_t cyclic = (x1 & ~z1 & x2 & z2)      // X*Y
                                     | (x1 & z1 & ~x2 & z2)    // Y*Z
                                     | (~x1 & z1 & x2 & ~z2);  // Z*X
        plus += static_cast<std::uint64_t>(std::popcount(cyclic));
        minus += static_cast<std::uint64_t>(std::popcount(anti & ~cyclic));
    }
    return static_cast<int>((plus + (3 * minus)) % 4);
}

std::array<std::complex<double>, 4> pauli_matrix_2x2(bool x, bool z) {
    using Complex = std::complex<double>;
    if (!x && !z) {
        return {Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}};
    }
    if (x && !z) {
        return {Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}};
    }
    if (!x && z) {
        return {Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{-1.0, 0.0}};
    }
    return {Complex{0.0, 0.0}, Complex{0.0, -1.0}, Complex{0.0, 1.0}, Complex{0.0, 0.0}};
}

}  // namespace menos::common
