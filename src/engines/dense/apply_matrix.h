#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

namespace menos::engines::dense {

// Applies a k-qubit gate matrix for any operand count k. Bit j of matrix's
// row/column index corresponds to operand operands[j] (CLAUDE.md's
// operand-order rule). Forwards to kernels::cpu::apply_matrix, which runs
// specialised kernels for k = 1 and k = 2 and the generic kernel otherwise,
// all bitwise identical to the generic one (docs/adr/0019 section 4).
//
// Precondition: matrix.size() == (1 << operands.size())^2; every entry of
// operands is a distinct, in-range bit position of state's amplitude
// index. Not re-checked here -- validated once at the KIR boundary
// (menos::ir::validate), not inside this kernel.
//
// This overload uses ONE thread. It is the reference the engine tests apply
// gates with; every engine's execution path calls the overload below with the
// job's thread count instead.
void apply_matrix(std::span<std::complex<double>> state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands);

// As above, on up to num_threads (>= 1) threads. The result is bitwise
// identical for every num_threads (docs/adr/0019 section 3.2).
void apply_matrix(std::span<std::complex<double>> state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands,
                  std::uint32_t num_threads);

// The heap bytes one apply_matrix call on k operands allocates while it runs:
// the generic kernel (k >= 3) gives each of its at most num_threads chunks a
// gather buffer of 2^k amplitudes and 2^k indices; the k = 1 and k = 2 kernels
// allocate nothing (docs/adr/0025 section 3.4).
[[nodiscard]] std::uint64_t apply_matrix_scratch_bytes(std::size_t k, std::uint32_t num_threads);

}  // namespace menos::engines::dense
