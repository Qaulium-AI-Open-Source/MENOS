#pragma once

// The CPU dense kernels' inner loops, as a table chosen once per process from
// the CPU's instruction set (Phase 24 item 8, docs/adr/0028 section 19).
//
// Two tables exist: the baseline one (plain x86-64, in dense_kernels.cpp) and
// the AVX2 one (dense_kernels_avx2.cpp, compiled with -mavx2 or /arch:AVX2 and
// never with FMA). active_dense_kernels() returns the AVX2 table when the CPU
// and the operating system support AVX2 and the baseline table otherwise.
// Every entry of the two tables computes, per amplitude, the same IEEE
// multiplies and adds in the same order, with no fused multiply-add, so the
// two are bitwise identical to each other and to apply_matrix_generic for the
// finite amplitudes and matrices the kernels are specified for.
//
// The entries take double pointers to interleaved (real, imaginary) storage,
// the layout std::complex<double> arrays are guaranteed to have, so that the
// AVX2 translation unit calls no inline function from a shared header (an
// out-of-line copy compiled with AVX2 could be the one the linker keeps for
// every caller).
//
// Internal to the kernels: nothing outside src/kernels and the kernel tests
// and probes includes this header.

#include <cstddef>
#include <cstdint>

namespace menos::kernels::cpu {

struct DenseKernelTable {
    // lo[i], hi[i] <- m * (lo[i], hi[i]) for i in [0, n): one 1-qubit gate on
    // n pairs whose members are hi - lo apart. m is the row-major 2x2 matrix.
    void (*pair_run)(double* lo, double* hi, std::size_t n, const double* m);
    // The same gate on operand q in {0, 1, 2, 3}, over every block of 2^(q+1)
    // amplitudes of amps[0, total).
    void (*pair_blocks)(double* amps, std::size_t total, std::uint32_t q, const double* m);
    // a[i] <- (+0, +0) + s * a[i] for i in [0, n).
    void (*scale_run)(double* a, std::size_t n, const double* s);
    // amps[i] <- (+0, +0) + d[local(i)] * amps[i] for i in [begin, end), local(i)
    // having bit j equal to bit operands[j] of i, j < k.
    void (*scale_elements)(double* amps,
                           std::size_t begin,
                           std::size_t end,
                           const std::uint32_t* operands,
                           std::size_t k,
                           const double* d);
    // The 2-qubit gate m (row-major 4x4) on operands (q0, q1), local bit 0
    // being q0, for the groups [begin, end) of amps: each output starts from
    // (+0, +0) and adds m[out][in] * a[in] for in = 0..3 in order.
    void (*two_qubit)(double* amps,
                      std::size_t begin,
                      std::size_t end,
                      std::uint32_t q0,
                      std::uint32_t q1,
                      const double* m);
};

enum class KernelIsa : std::uint8_t { Baseline, Avx2 };

// The plain x86-64 table.
const DenseKernelTable& baseline_dense_kernels();
// The AVX2 table, or nullptr in a build without it (a non-x86-64 target).
const DenseKernelTable* avx2_dense_kernels();
// Whether this CPU and operating system can run the AVX2 table.
bool cpu_supports_avx2();
// The table apply_matrix uses: AVX2 where supported, else baseline, unless a
// test has forced one with force_dense_kernel_isa.
const DenseKernelTable& active_dense_kernels();
KernelIsa active_dense_kernel_isa();

// For the kernel tests and probes only: makes active_dense_kernels() return
// the given ISA's table (Avx2 requires cpu_supports_avx2() and an AVX2 build)
// until reset_dense_kernel_isa(). Not thread-safe against concurrent kernels.
void force_dense_kernel_isa(KernelIsa isa);
void reset_dense_kernel_isa();

}  // namespace menos::kernels::cpu
