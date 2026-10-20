#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/error.h"
#include "kernels/cuda/cuda_error.h"
#include "kernels/dense_kernels.h"
#include "menos/circuit.h"

namespace menos::kernels::cuda {
namespace {

constexpr int kBlockSize = 256;

// kMaxUnitaryQubits (5) is the largest operand count menos::ir::validate
// admits, and the named gate table tops out at 3 (CCX). So this bound is not a
// CUDA limitation that could reject a valid circuit -- it is the same ceiling
// the IR already enforces, restated here because the kernel needs a
// compile-time array size. static_assert keeps the two from drifting apart.
constexpr int kMaxOperands = 5;
static_assert(kMaxOperands == menos::kMaxUnitaryQubits,
              "the device kernel's operand ceiling must track kMaxUnitaryQubits");
constexpr int kMaxMatrixEntries = 1 << (2 * kMaxOperands);  // (2^k)^2 at k = 5

// The gate matrix and operand list live in __constant__ memory rather than
// being re-allocated per gate. Both reasons are measured, not stylistic:
// cudaMalloc/cudaFree per gate costs single-digit microseconds, which is
// nothing against a large state but dominates a small one and would push the
// CPU/GPU crossover right for no reason; and every thread reads the same
// matrix, which is exactly the broadcast access pattern constant memory is
// for. 16 KiB at k = 5, within the 64 KiB constant bank.
//
// PRECONDITION, stated because it is a real constraint and not enforced by the
// type system: these are device-module-global, so apply_matrix must not be
// called concurrently from two host threads against the same device. MENOS has
// no threaded execution yet (CLAUDE.md requires a ThreadSanitizer preset to
// arrive with the first threaded code, and none exists), so this holds today;
// it must be revisited when that changes.
__constant__ cuDoubleComplex c_matrix[kMaxMatrixEntries];
__constant__ std::uint32_t c_operands[kMaxOperands];

__device__ inline cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) {
    // Written with explicit real/imaginary arithmetic, per CLAUDE.md's hot
    // kernel rule.
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

// One thread per outer (compact) index. Mirrors the CPU primitive's structure
// exactly -- expand, gather 2^k, multiply, scatter -- so the two produce the
// same amplitudes, not merely similar ones.
__global__ void apply_matrix_kernel(cuDoubleComplex* __restrict__ state,
                                    std::size_t n_outer,
                                    int num_qubits,
                                    int k,
                                    std::uint64_t operand_mask) {
    const std::size_t compact = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact >= n_outer) {
        return;
    }

    const std::size_t dim = std::size_t{1} << k;

    // Expand compact's (n-k) bits into an n-bit base index, leaving zeros at
    // the operand positions.
    std::size_t base = 0;
    int bit_src = 0;
    for (int p = 0; p < num_qubits; ++p) {
        if ((operand_mask >> p) & std::uint64_t{1}) {
            continue;
        }
        base |= ((compact >> bit_src) & std::size_t{1}) << p;
        ++bit_src;
    }

    std::size_t full_indices[1 << kMaxOperands];
    cuDoubleComplex gathered[1 << kMaxOperands];
    for (std::size_t local = 0; local < dim; ++local) {
        std::size_t idx = base;
        for (int j = 0; j < k; ++j) {
            if ((local >> j) & std::size_t{1}) {
                idx |= (std::size_t{1} << c_operands[j]);
            }
        }
        full_indices[local] = idx;
        gathered[local] = state[idx];
    }

    for (std::size_t out = 0; out < dim; ++out) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (std::size_t in = 0; in < dim; ++in) {
            sum = cadd(sum, cmul(c_matrix[out * dim + in], gathered[in]));
        }
        state[full_indices[out]] = sum;
    }
}

// The specialised k = 1 and k = 2 kernels (docs/adr/0020 section 3) take the
// matrix by value as a kernel argument, so apply_matrix makes no
// cudaMemcpyToSymbol for them: those two copies were the larger part of a
// small gate's cost (docs/adr/0020 section 10.3). Kernel arguments are
// broadcast to every thread through the constant bank, as c_matrix is.
struct Matrix2 {
    cuDoubleComplex m[4];
};
struct Matrix4 {
    cuDoubleComplex m[16];
};
static_assert(sizeof(Matrix2) == 4 * sizeof(cuDoubleComplex) &&
                  sizeof(Matrix4) == 16 * sizeof(cuDoubleComplex),
              "the matrix arguments must be the packed row-major matrices");

// i with a zero bit inserted at position pos: the bits at and above pos move
// up by one. The same helper as the CPU's specialised kernels.
__device__ inline std::size_t insert_zero_bit(std::size_t i, std::uint32_t pos) {
    const std::size_t low = (std::size_t{1} << pos) - 1;
    return ((i & ~low) << 1) | (i & low);
}

// k = 1, one thread per amplitude pair (i0, i1) differing only in bit q. The
// generic kernel gathers {state[i0], state[i1]} as local indices 0 and 1 and
// computes out[r] = (0 + m[r][0] * a0) + m[r][1] * a1 with the same cmul and
// cadd. This computes the same expression in the same order, so the result
// is BITWISE identical; only the index arithmetic differs.
__global__ void apply_matrix_1q_kernel(cuDoubleComplex* __restrict__ state,
                                       std::size_t n_pairs,
                                       std::uint32_t q,
                                       Matrix2 matrix) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_pairs) {
        return;
    }
    const std::size_t i0 = insert_zero_bit(i, q);
    const std::size_t i1 = i0 | (std::size_t{1} << q);
    const cuDoubleComplex a0 = state[i0];
    const cuDoubleComplex a1 = state[i1];
    cuDoubleComplex out0 = make_cuDoubleComplex(0.0, 0.0);
    out0 = cadd(out0, cmul(matrix.m[0], a0));
    out0 = cadd(out0, cmul(matrix.m[1], a1));
    cuDoubleComplex out1 = make_cuDoubleComplex(0.0, 0.0);
    out1 = cadd(out1, cmul(matrix.m[2], a0));
    out1 = cadd(out1, cmul(matrix.m[3], a1));
    state[i0] = out0;
    state[i1] = out1;
}

// k = 2, one thread per group of four. Local index bit j is operand j
// (q0 is local bit 0), exactly as in the generic kernel, so a reversed or
// non-adjacent pair keeps the generic kernel's local ordering. Zero bits are
// inserted at lo = min(q0, q1) first, then hi, which leaves both positions
// clear. Each output sums in = 0..3 in order from zero: bitwise identical.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
__global__ void apply_matrix_2q_kernel(cuDoubleComplex* __restrict__ state,
                                       std::size_t n_groups,
                                       std::uint32_t q0,
                                       std::uint32_t q1,
                                       Matrix4 matrix) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_groups) {
        return;
    }
    const std::uint32_t lo = q0 < q1 ? q0 : q1;
    const std::uint32_t hi = q0 < q1 ? q1 : q0;
    const std::size_t b0 = std::size_t{1} << q0;
    const std::size_t b1 = std::size_t{1} << q1;
    const std::size_t base = insert_zero_bit(insert_zero_bit(i, lo), hi);
    const std::size_t idx[4] = {base, base | b0, base | b1, base | b0 | b1};
    const cuDoubleComplex a[4] = {state[idx[0]], state[idx[1]], state[idx[2]], state[idx[3]]};
#pragma unroll
    for (int out = 0; out < 4; ++out) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
#pragma unroll
        for (int in = 0; in < 4; ++in) {
            sum = cadd(sum, cmul(matrix.m[out * 4 + in], a[in]));
        }
        state[idx[out]] = sum;
    }
}

// Per-block partial sums of conj(psi[m]) * psi[m^x] * (-1)^popcount(z & (m^x)),
// docs/adr/0007's single-pass formula. The i^popcount(x&z) prefactor is applied
// once on the host, not per element.
__global__ void expectation_kernel(const cuDoubleComplex* __restrict__ state,
                                   std::size_t num_amplitudes,
                                   std::uint64_t x_mask,
                                   std::uint64_t z_mask,
                                   cuDoubleComplex* __restrict__ partials) {
    __shared__ double shared_real[kBlockSize];
    __shared__ double shared_imag[kBlockSize];

    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double acc_real = 0.0;
    double acc_imag = 0.0;

    for (std::size_t m = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         m < num_amplitudes;
         m += stride) {
        const std::uint64_t k = static_cast<std::uint64_t>(m) ^ x_mask;
        const cuDoubleComplex a = state[m];
        const cuDoubleComplex b = state[k];
        // conj(a) * b, explicit real/imaginary arithmetic.
        double real = a.x * b.x + a.y * b.y;
        double imag = a.x * b.y - a.y * b.x;
        if ((__popcll(z_mask & k) & 1) != 0) {
            real = -real;
            imag = -imag;
        }
        acc_real += real;
        acc_imag += imag;
    }

    shared_real[threadIdx.x] = acc_real;
    shared_imag[threadIdx.x] = acc_imag;
    __syncthreads();

    // Fixed-order tree reduction: the summation order is a deterministic
    // function of the block size alone, so the same input gives the same output
    // on every run. It differs from the CPU's sequential order, which is why
    // the two agree to tolerance rather than bitwise (docs/adr/0014).
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_real[threadIdx.x] += shared_real[threadIdx.x + s];
            shared_imag[threadIdx.x] += shared_imag[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = make_cuDoubleComplex(shared_real[0], shared_imag[0]);
    }
}

// i^k for k in {0,1,2,3}, matching the CPU primitive's helper exactly.
std::complex<double> i_power(int k) {
    switch (k & 3) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
    }
}

std::size_t grid_for(std::size_t items) {
    return (items + kBlockSize - 1) / kBlockSize;
}

void launch_and_synchronize_check(const char* kernel_name) {
    ensure_cuda(cudaGetLastError(), kernel_name);
    // One synchronisation per gate, kept (docs/adr/0020 Q4): the Phase 11
    // monitor reads the clock at each gate boundary and relies on the gate
    // having finished on the device (src/engines/dense_cuda/cuda_dense_engine.cpp).
    ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}  // namespace

void initialize(DeviceState state, double global_phase) {
    common::ensure(state.num_amplitudes > 0, "cuda::initialize: state must be non-empty");
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(state.num_amplitudes) * sizeof(std::complex<double>);
    ensure_cuda(cudaMemset(state.data, 0, bytes), "cudaMemset");

    // exp(i * global_phase) into element 0. Computed on the host with the same
    // std::exp the CPU primitive uses, so the two starting states are bitwise
    // identical rather than merely close.
    const std::complex<double> phase = std::exp(std::complex<double>{0.0, global_phase});
    ensure_cuda(cudaMemcpy(state.data, &phase, sizeof(phase), cudaMemcpyHostToDevice),
                "cudaMemcpy host-to-device (global phase)");
    ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void apply_matrix(DeviceState state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands) {
    // std::complex<double> and cuDoubleComplex share a layout (two doubles,
    // real first), the same assumption the generic path's copy into c_matrix
    // makes; copied element-wise here so no pointer is reinterpreted.
    const auto to_device = [](const std::complex<double>& z) {
        return make_cuDoubleComplex(z.real(), z.imag());
    };
    switch (operands.size()) {
        case 1: {
            common::ensure(matrix.size() == 4,
                           "cuda::apply_matrix: matrix size does not match 2^|operands|");
            Matrix2 m{};
            for (std::size_t e = 0; e < 4; ++e) {
                m.m[e] = to_device(matrix[e]);
            }
            const std::size_t n_pairs = state.num_amplitudes >> 1;
            apply_matrix_1q_kernel<<<static_cast<unsigned int>(grid_for(n_pairs)), kBlockSize>>>(
                static_cast<cuDoubleComplex*>(state.data), n_pairs, operands[0], m);
            launch_and_synchronize_check("apply_matrix_1q_kernel launch");
            return;
        }
        case 2: {
            common::ensure(matrix.size() == 16,
                           "cuda::apply_matrix: matrix size does not match 2^|operands|");
            Matrix4 m{};
            for (std::size_t e = 0; e < 16; ++e) {
                m.m[e] = to_device(matrix[e]);
            }
            const std::size_t n_groups = state.num_amplitudes >> 2;
            apply_matrix_2q_kernel<<<static_cast<unsigned int>(grid_for(n_groups)), kBlockSize>>>(
                static_cast<cuDoubleComplex*>(state.data), n_groups, operands[0], operands[1], m);
            launch_and_synchronize_check("apply_matrix_2q_kernel launch");
            return;
        }
        default:
            apply_matrix_generic(state, matrix, operands);
            return;
    }
}

void apply_matrix_generic(DeviceState state,
                          std::span<const std::complex<double>> matrix,
                          std::span<const std::uint32_t> operands) {
    const std::size_t k = operands.size();
    const std::size_t dim = std::size_t{1} << k;
    common::ensure(matrix.size() == dim * dim,
                   "cuda::apply_matrix: matrix size does not match 2^|operands|");
    common::ensure(k >= 1 && k <= static_cast<std::size_t>(kMaxOperands),
                   "cuda::apply_matrix: operand count outside [1, kMaxUnitaryQubits]");

    const std::size_t total = state.num_amplitudes;
    int num_qubits = 0;
    while ((std::size_t{1} << num_qubits) < total) {
        ++num_qubits;
    }

    std::uint64_t operand_mask = 0;
    for (std::uint32_t q : operands) {
        operand_mask |= (std::uint64_t{1} << q);
    }

    ensure_cuda(
        cudaMemcpyToSymbol(c_matrix, matrix.data(), matrix.size() * sizeof(std::complex<double>)),
        "cudaMemcpyToSymbol (matrix)");
    ensure_cuda(cudaMemcpyToSymbol(c_operands, operands.data(), k * sizeof(std::uint32_t)),
                "cudaMemcpyToSymbol (operands)");

    const std::size_t n_outer = total >> k;
    apply_matrix_kernel<<<static_cast<unsigned int>(grid_for(n_outer)), kBlockSize>>>(
        static_cast<cuDoubleComplex*>(state.data),
        n_outer,
        num_qubits,
        static_cast<int>(k),
        operand_mask);
    launch_and_synchronize_check("apply_matrix_kernel launch");
}

std::complex<double> expectation_of_term(ConstDeviceState state,
                                         std::uint64_t x_mask,
                                         std::uint64_t z_mask) {
    const std::size_t total = state.num_amplitudes;
    common::ensure(total > 0, "cuda::expectation_of_term: state must be non-empty");

    // Bounded grid: each block accumulates over a strided range, so the partial
    // array stays small enough to sum on the host in fixed order regardless of
    // qubit count.
    const std::size_t blocks = std::min(kExpectationMaxBlocks, grid_for(total));

    void* partials_device = nullptr;
    const std::size_t partial_bytes = blocks * sizeof(std::complex<double>);
    ensure_cuda(cudaMalloc(&partials_device, partial_bytes), "cudaMalloc (partials)");

    expectation_kernel<<<static_cast<unsigned int>(blocks), kBlockSize>>>(
        static_cast<const cuDoubleComplex*>(state.data),
        total,
        x_mask,
        z_mask,
        static_cast<cuDoubleComplex*>(partials_device));
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        static_cast<void>(cudaFree(partials_device));
        ensure_cuda(launch_status, "expectation_kernel launch");
    }

    std::vector<std::complex<double>> partials(blocks);
    const cudaError_t copy_status =
        cudaMemcpy(partials.data(), partials_device, partial_bytes, cudaMemcpyDeviceToHost);
    static_cast<void>(cudaFree(partials_device));
    ensure_cuda(copy_status, "cudaMemcpy device-to-host (partials)");

    // Summed on the host in ascending block order: a fixed, device-independent
    // order, so the result does not depend on how many blocks the launch chose.
    std::complex<double> sum{0.0, 0.0};
    for (const std::complex<double>& partial : partials) {
        sum += partial;
    }
    return i_power(__builtin_popcountll(x_mask & z_mask)) * sum;
}

}  // namespace menos::kernels::cuda
