#include "engines/stabilizer/tableau.h"

#include <utility>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/memory_ledger.h"

namespace menos::engines::stabilizer {
namespace {

using common::get_bit;
using common::PauliRow;
using common::set_bit;

// row_h <- row_h * row_i (Pauli operator product), docs/adr/0006's
// rowsum: XOR the bits, correct the sign via the accumulated i-power
// (must reduce to a real +-1, enforced below). The phase is summed a word
// at a time (docs/adr/0028 section 22): until Phase 24 it was summed one
// qubit at a time, which made a rowsum O(n) rather than O(n / 64) and the
// eliminations O(n^3) on a dense tableau.
void rowsum(PauliRow& h, const PauliRow& i) {
    int total = (h.r ? 2 : 0) + (i.r ? 2 : 0) + common::pauli_product_phase_exponent(h, i);
    total %= 4;
    common::ensure(total == 0 || total == 2,
                   "Tableau::rowsum: accumulated phase is not real -- internal invariant violated");
    h.r = (total == 2);
    for (std::size_t w = 0; w < h.x.size(); ++w) {
        h.x[w] ^= i.x[w];
        h.z[w] ^= i.z[w];
    }
}

// One established pivot of expectation_sign()'s elimination.
struct Pivot {
    bool is_z;
    std::uint32_t col;
};

std::uint64_t words_of(std::uint32_t num_qubits) {
    return (std::uint64_t{num_qubits} + 63) / 64;
}

// One allocation of `count` words.
std::uint64_t words_block(std::uint64_t count) {
    return common::heap_block_bytes(common::checked_mul(count, sizeof(std::uint64_t)));
}

// `count` PauliRows of n qubits in one std::vector: the array, and each row's
// x and z words, two allocations per row.
std::uint64_t rows_array_bytes(std::uint64_t count, std::uint32_t num_qubits) {
    return common::checked_add(
        common::heap_block_bytes(common::checked_mul(count, sizeof(PauliRow))),
        common::checked_mul(2 * count, words_block(words_of(num_qubits))));
}

// An n-bit std::vector<bool>'s allocation: at most one 64-bit word per 64 bits
// (libstdc++ packs into 64-bit words, the MSVC STL into 32-bit ones).
std::uint64_t bool_vector_bytes(std::uint32_t num_qubits) {
    return words_block(words_of(num_qubits));
}

}  // namespace

std::uint64_t Tableau::rows_bytes(std::uint32_t num_qubits) {
    return rows_array_bytes(num_qubits, num_qubits);
}

std::uint64_t Tableau::subspace_bytes(std::uint32_t num_qubits) {
    // base, and up to n generators of n bits each in one reserved list.
    return common::checked_add(
        bool_vector_bytes(num_qubits),
        common::checked_add(
            common::heap_block_bytes(common::checked_mul(num_qubits, sizeof(std::vector<bool>))),
            common::checked_mul(num_qubits, bool_vector_bytes(num_qubits))));
}

std::uint64_t Tableau::reduce_peak_bytes(std::uint32_t num_qubits) {
    const std::uint64_t n = num_qubits;
    // work (n rows) and fixed (at most n rows), the pivot-column list and
    // flags, free_cols (grown one push_back at a time), and the subspace.
    std::uint64_t bytes = common::checked_mul(2, rows_array_bytes(n, num_qubits));
    bytes = common::checked_add(bytes, common::heap_block_bytes(n * sizeof(std::uint32_t)));
    bytes = common::checked_add(bytes, bool_vector_bytes(num_qubits));
    bytes = common::checked_add(bytes, common::grown_vector_bytes(n, sizeof(std::uint32_t)));
    return common::checked_add(bytes, subspace_bytes(num_qubits));
}

std::uint64_t Tableau::expectation_peak_bytes(std::uint32_t num_qubits) {
    const std::uint64_t n = num_qubits;
    const std::uint64_t word_bytes = words_block(words_of(num_qubits));
    // work, masks (n vectors of n bits in one list), pivots (reserved to n),
    // the residual (a copy of the query's two word lists), combo and product.
    std::uint64_t bytes = rows_array_bytes(n, num_qubits);
    bytes = common::checked_add(bytes,
                                common::checked_add(common::heap_block_bytes(common::checked_mul(
                                                        n, sizeof(std::vector<std::uint64_t>))),
                                                    common::checked_mul(n, word_bytes)));
    bytes = common::checked_add(bytes, common::heap_block_bytes(n * sizeof(Pivot)));
    return common::checked_add(bytes, 5 * word_bytes);
}

namespace {

// The bytes a list of rows asked for: the list and each row's words.
std::uint64_t rows_held_bytes(const std::vector<PauliRow>& rows) {
    std::uint64_t bytes = common::buffer_bytes(rows);
    for (const PauliRow& row : rows) {
        bytes += common::buffer_bytes(row.x) + common::buffer_bytes(row.z);
    }
    return bytes;
}

}  // namespace

std::uint64_t Tableau::held_bytes() const {
    return rows_held_bytes(rows_);
}

Tableau::Tableau(std::uint32_t num_qubits) : num_qubits_(num_qubits) {
    common::ensure(num_qubits >= 1, "Tableau: num_qubits must be at least 1");
    const std::size_t words = words_per_row();
    rows_.resize(num_qubits);
    for (std::uint32_t i = 0; i < num_qubits; ++i) {
        rows_[i].x.assign(words, 0);
        rows_[i].z.assign(words, 0);
        set_bit(rows_[i].z, i, true);  // row i = Z_i, r = 0: |0...0>'s stabilizers
        rows_[i].r = false;
    }
}

std::size_t Tableau::words_per_row() const {
    return (static_cast<std::size_t>(num_qubits_) + 63) / 64;
}

std::uint32_t Tableau::num_qubits() const {
    return num_qubits_;
}

void Tableau::apply_x(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_x(row, a);
    }
}

void Tableau::apply_z(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_z(row, a);
    }
}

void Tableau::apply_y(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_y(row, a);
    }
}

void Tableau::apply_h(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_h(row, a);
    }
}

void Tableau::apply_s(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_s(row, a);
    }
}

void Tableau::apply_sdg(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_sdg(row, a);
    }
}

void Tableau::apply_sx(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_sx(row, a);
    }
}

void Tableau::apply_sxdg(std::uint32_t a) {
    for (PauliRow& row : rows_) {
        common::apply_sxdg(row, a);
    }
}

void Tableau::apply_cx(std::uint32_t control, std::uint32_t target) {
    for (PauliRow& row : rows_) {
        common::apply_cx(row, control, target);
    }
}

void Tableau::apply_cz(std::uint32_t a, std::uint32_t b) {
    for (PauliRow& row : rows_) {
        common::apply_cz(row, a, b);
    }
}

void Tableau::apply_swap(std::uint32_t a, std::uint32_t b) {
    for (PauliRow& row : rows_) {
        common::apply_swap(row, a, b);
    }
}

Tableau::AffineSubspace Tableau::reduce() const {
    const std::uint32_t n = num_qubits_;
    std::vector<PauliRow> work = rows_;
    const common::MemoryLedger::Charge work_charge(rows_held_bytes(work));

    // Pass 1: Gauss-Jordan eliminate the x-part. Rows [0, r_dim) end up
    // as the pivot rows (nonzero x); rows [r_dim, n) are forced to
    // x == 0 everywhere (a basic property of row-reduced echelon form:
    // a rank-r_dim matrix has exactly n - r_dim all-zero rows after
    // full reduction, regardless of which columns become pivots).
    std::size_t pivot_row = 0;
    for (std::uint32_t col = 0; col < n && pivot_row < n; ++col) {
        std::size_t sel = n;
        for (std::size_t cand = pivot_row; cand < n; ++cand) {
            if (get_bit(work[cand].x, col)) {
                sel = cand;
                break;
            }
        }
        if (sel == n) {
            continue;
        }
        std::swap(work[pivot_row], work[sel]);
        for (std::size_t cand = 0; cand < n; ++cand) {
            if (cand != pivot_row && get_bit(work[cand].x, col)) {
                rowsum(work[cand], work[pivot_row]);
            }
        }
        ++pivot_row;
    }
    const std::size_t r_dim = pivot_row;

    std::vector<PauliRow> fixed(work.begin() + static_cast<std::ptrdiff_t>(r_dim), work.end());
    const common::MemoryLedger::Charge fixed_charge(rows_held_bytes(fixed));
    const std::size_t num_fixed = fixed.size();
    common::ensure(num_fixed == static_cast<std::size_t>(n) - r_dim,
                   "Tableau::reduce: fixed-row count does not match n - r_dim");

    // Pass 2: Gauss-Jordan eliminate the fixed rows' z-part (their
    // x-part is identically zero, so rowsum's phase correction always
    // contributes 0 here -- this is plain GF(2) elimination). The
    // fixed rows are guaranteed independent (their (x=0, z) vectors
    // inherit independence from the original n independent generators),
    // so every one of them becomes a pivot row.
    std::vector<uint32_t> fixed_pivot_col(num_fixed, n);
    std::vector<bool> is_pivot_col(n, false);
    std::size_t fp_row = 0;
    for (std::uint32_t col = 0; col < n && fp_row < num_fixed; ++col) {
        std::size_t sel = num_fixed;
        for (std::size_t cand = fp_row; cand < num_fixed; ++cand) {
            if (get_bit(fixed[cand].z, col)) {
                sel = cand;
                break;
            }
        }
        if (sel == num_fixed) {
            continue;
        }
        std::swap(fixed[fp_row], fixed[sel]);
        for (std::size_t cand = 0; cand < num_fixed; ++cand) {
            if (cand != fp_row && get_bit(fixed[cand].z, col)) {
                rowsum(fixed[cand], fixed[fp_row]);
            }
        }
        fixed_pivot_col[fp_row] = col;
        is_pivot_col[col] = true;
        ++fp_row;
    }
    common::ensure(
        fp_row == num_fixed,
        "Tableau::reduce: fixed rows are not independent -- internal invariant violated");

    std::vector<std::uint32_t> free_cols;
    for (std::uint32_t col = 0; col < n; ++col) {
        if (!is_pivot_col[col]) {
            free_cols.push_back(col);
        }
    }
    common::ensure(free_cols.size() == r_dim,
                   "Tableau::reduce: free-column count does not match the affine dimension");

    std::vector<bool> base(n, false);
    for (std::size_t j = 0; j < num_fixed; ++j) {
        base[fixed_pivot_col[j]] = fixed[j].r;
    }

    std::vector<std::vector<bool>> generators;
    generators.reserve(free_cols.size());
    for (std::uint32_t f : free_cols) {
        std::vector<bool> g(n, false);
        g[f] = true;
        for (std::size_t j = 0; j < num_fixed; ++j) {
            g[fixed_pivot_col[j]] = get_bit(fixed[j].z, f);
        }
        generators.push_back(std::move(g));
    }

    return AffineSubspace{std::move(base), std::move(generators)};
}

// docs/adr/0007: the n stabilizer generators span an n-dimensional
// Lagrangian subspace of GF(2)^{2n} under the symplectic form, so
// "commutes with every generator" and "is a GF(2) linear combination of
// the generators" are the same condition. Gauss-Jordan eliminate the n
// generators' full (x,z) vectors while tracking, per pivot, an n-bit
// mask of which ORIGINAL rows combine to reconstruct it (the standard
// "augment with an identity matrix" change-of-basis technique); reduce
// the query against the same pivots in establishment order, then either
// report a zero residual (combo names the exact original-generator
// subset, rowsum them for the sign) or a nonzero one (anticommutes with
// some generator, expectation 0).
std::optional<bool> Tableau::expectation_sign(const PauliString& query) const {
    const std::uint32_t n = num_qubits_;
    common::ensure(query.num_qubits == n,
                   "Tableau::expectation_sign: query qubit count does not match this tableau");

    std::vector<PauliRow> work = rows_;
    const std::size_t words = words_per_row();
    std::vector<std::vector<std::uint64_t>> masks(n, std::vector<std::uint64_t>(words, 0));
    const common::MemoryLedger::Charge work_charge(
        rows_held_bytes(work) + common::buffer_bytes(masks) +
        std::uint64_t{n} * words * sizeof(std::uint64_t));
    for (std::uint32_t i = 0; i < n; ++i) {
        set_bit(masks[i], i, true);
    }

    std::vector<Pivot> pivots;
    pivots.reserve(n);

    std::size_t pivot_row = 0;
    for (std::uint32_t col = 0; col < n && pivot_row < n; ++col) {
        std::size_t sel = n;
        for (std::size_t cand = pivot_row; cand < n; ++cand) {
            if (get_bit(work[cand].x, col)) {
                sel = cand;
                break;
            }
        }
        if (sel == n) {
            continue;
        }
        std::swap(work[pivot_row], work[sel]);
        std::swap(masks[pivot_row], masks[sel]);
        for (std::size_t cand = 0; cand < n; ++cand) {
            if (cand != pivot_row && get_bit(work[cand].x, col)) {
                rowsum(work[cand], work[pivot_row]);
                for (std::size_t w = 0; w < words; ++w) {
                    masks[cand][w] ^= masks[pivot_row][w];
                }
            }
        }
        pivots.push_back(Pivot{false, col});
        ++pivot_row;
    }
    for (std::uint32_t col = 0; col < n && pivot_row < n; ++col) {
        std::size_t sel = n;
        for (std::size_t cand = pivot_row; cand < n; ++cand) {
            if (get_bit(work[cand].z, col)) {
                sel = cand;
                break;
            }
        }
        if (sel == n) {
            continue;
        }
        std::swap(work[pivot_row], work[sel]);
        std::swap(masks[pivot_row], masks[sel]);
        for (std::size_t cand = 0; cand < n; ++cand) {
            if (cand != pivot_row && get_bit(work[cand].z, col)) {
                rowsum(work[cand], work[pivot_row]);
                for (std::size_t w = 0; w < words; ++w) {
                    masks[cand][w] ^= masks[pivot_row][w];
                }
            }
        }
        pivots.push_back(Pivot{true, col});
        ++pivot_row;
    }
    common::ensure(pivot_row == n,
                   "Tableau::expectation_sign: generators are not full rank -- internal "
                   "invariant violated");

    PauliRow residual{query.x_words, query.z_words, false};
    std::vector<std::uint64_t> combo(words, 0);
    for (std::size_t i = 0; i < pivots.size(); ++i) {
        const bool bit = pivots[i].is_z ? get_bit(residual.z, pivots[i].col)
                                        : get_bit(residual.x, pivots[i].col);
        if (bit) {
            for (std::size_t w = 0; w < words; ++w) {
                residual.x[w] ^= work[i].x[w];
                residual.z[w] ^= work[i].z[w];
                combo[w] ^= masks[i][w];
            }
        }
    }

    bool residual_zero = true;
    for (std::size_t w = 0; w < words; ++w) {
        if (residual.x[w] != 0 || residual.z[w] != 0) {
            residual_zero = false;
            break;
        }
    }
    if (!residual_zero) {
        return std::nullopt;  // anticommutes with some generator: expectation 0
    }

    PauliRow product{
        std::vector<std::uint64_t>(words, 0), std::vector<std::uint64_t>(words, 0), false};
    for (std::uint32_t j = 0; j < n; ++j) {
        if (get_bit(combo, j)) {
            rowsum(product, rows_[j]);
        }
    }
    common::ensure(product.x == query.x_words && product.z == query.z_words,
                   "Tableau::expectation_sign: reconstructed generator product does not match "
                   "the query -- internal invariant violated");
    return product.r;
}

}  // namespace menos::engines::stabilizer
