#include "engines/dense/apply_matrix.h"

#include "common/checked_math.h"
#include "common/memory_bounds.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense {

// The body moved to the primitive API when the CUDA backend arrived
// (docs/adr/0014). These forwarders are kept so the dense engine's existing
// call sites and its oracle tests (tests/engines/dense/apply_matrix_test.cpp)
// keep exercising the same code through the same name.
void apply_matrix(std::span<std::complex<double>> state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands) {
    kernels::cpu::apply_matrix(state, matrix, operands, /*num_threads=*/1);
}

void apply_matrix(std::span<std::complex<double>> state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands,
                  std::uint32_t num_threads) {
    kernels::cpu::apply_matrix(state, matrix, operands, num_threads);
}

// An operand count and a thread count: distinct roles, named at every call.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::uint64_t apply_matrix_scratch_bytes(std::size_t k, std::uint32_t num_threads) {
    if (k <= 2) {
        return 0;
    }
    const std::uint64_t entries = common::checked_pow2(static_cast<unsigned>(k));
    const std::uint64_t per_chunk = common::checked_add(
        common::heap_block_bytes(common::checked_mul(entries, sizeof(std::size_t))),
        common::heap_block_bytes(common::checked_mul(entries, sizeof(std::complex<double>))));
    return common::checked_mul(per_chunk, num_threads);
}

}  // namespace menos::engines::dense
