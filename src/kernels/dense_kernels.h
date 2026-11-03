#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// The dense-statevector primitive API (docs/BUILD_PLAN.md Phase 10,
// docs/adr/0014).
//
// These are the THREE operations the CPU and CUDA dense backends genuinely
// share, and nothing else. CLAUDE.md prime directive 3 is why the list is this
// short and why this header did not exist before: "the kernel primitive API
// arrives with the second hardware backend". It was extracted from the two
// concrete implementations, not designed ahead of them, so every declaration
// here has two callers.
//
// It is deliberately NOT a virtual interface. CLAUDE.md's architecture rule
// says virtual dispatch happens at region or engine granularity, never per gate
// inside a hot loop, so the backends are selected once (DenseEngine calls
// kernels::cpu, CudaDenseEngine calls kernels::cuda) and every gate afterwards
// is a direct call.
//
// ---------------------------------------------------------------------------
// The conventions both implementations must obey, stated once, here
// ---------------------------------------------------------------------------
//
// Qubit order is little-endian: bit j of an amplitude index is the state of
// qubit j. This matches Qiskit and is never reversed internally.
//
// Operand order: for operands (a0, ..., a_{k-1}), bit j of the gate matrix's
// row and column index corresponds to operand a_j. CX(control, target) has
// operands (control, target). An implementation realises this by zero-bit
// insertion at the operand positions: each outer index enumerates the
// non-operand bits, and the 2^k operand-bit patterns are gathered, multiplied
// by the matrix and scattered back.
//
// Amplitudes are complex128 everywhere.
//
// Global phase is part of the state, not a bookkeeping extra: a circuit's
// statevector starts at exp(i * global_phase) |0...0>, so RZ(t) and P(t) are
// distinguishable outputs.
//
// Pauli phase convention is Y = i * X * Z, matching Qiskit's Pauli /
// SparsePauliOp.
//
// ---------------------------------------------------------------------------
// What is NOT shared, and why it is not here
// ---------------------------------------------------------------------------
//
// Counts sampling is absent on purpose. CLAUDE.md requires the same circuit,
// options and seed to produce the same counts on every supported platform, and
// a device-side sampler would not reproduce the CPU sampler's draw sequence.
// The CUDA backend therefore copies its state back and uses the existing CPU
// sampler, so counts are bit-identical between backends rather than merely
// close. A sampling primitive would have exactly one real implementation, which
// prime directive 2 forbids.

namespace menos::kernels {

// A dense statevector living in device memory, described without naming any
// CUDA type so that this header stays compilable on a machine with no CUDA
// toolkit at all. data points at num_amplitudes contiguous complex128 values
// in device address space; it is never dereferenced by host code.
struct DeviceState {
    void* data;
    std::size_t num_amplitudes;
};

struct ConstDeviceState {
    const void* data;
    std::size_t num_amplitudes;

    ConstDeviceState(const void* device_data, std::size_t amplitudes)
        : data(device_data), num_amplitudes(amplitudes) {}

    // Implicit, mirroring std::span<T> -> std::span<const T>: a mutable device
    // state is usable wherever a read-only one is wanted, so a caller holding a
    // DeviceBuffer does not have to spell the conversion out.
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    ConstDeviceState(DeviceState state) : data(state.data), num_amplitudes(state.num_amplitudes) {}
};

namespace cpu {

// A gate with fewer independent amplitude groups than this (2^(n - k) for a
// k-qubit gate on n qubits) runs on the calling thread alone; from this size
// on it runs on the whole team, num_threads. MEASURED end to end on the Phase
// 15 gate machine (docs/adr/0019 section 10): waking the pool costs about
// 200 us whatever the team size, and on the dashboard circuit the threaded run
// first matched the single-threaded one at 17 qubits (1-qubit gates 2^16
// groups, 2-qubit gates 2^15) and beat it from 18 (2^16 and 2^17 groups:
// 1.83x). A kernel-only measurement had suggested 2^14, but its single-thread
// times were inflated by the thread landing on an efficiency core; on the
// real circuit 2^14-2^15 groups ran slower threaded. It decides only speed:
// every group is independent, so results are bitwise identical for every team
// size. Public so tests can make sure their circuits are threaded.
inline constexpr std::size_t kParallelMinGroups = std::size_t{1} << 16;

// Sets state to exp(i * global_phase) |0...0>.
// Precondition: state is non-empty.
void initialize(std::span<std::complex<double>> state, double global_phase);

// Applies matrix to state at the given operands, in place.
// Precondition: matrix.size() == (1 << operands.size())^2; every operand is a
// distinct, in-range bit position of state's amplitude index. Not re-checked
// per call -- validated once at the KIR boundary (menos::ir::validate), never
// inside the kernel (CLAUDE.md: validate at module boundaries, not in kernels).
// Complexity: O(2^(n + k)) for k = operands.size().
//
// k = 1 and k = 2 run specialised kernels (docs/adr/0019 section 4); k >= 3
// runs apply_matrix_generic. A matrix whose off-diagonal entries are all
// exactly zero (RZ, RZZ, CP, a fused diagonal block) takes, for every k, a
// streaming diagonal kernel: one complex multiply per amplitude and no
// gather, computing `(+0, +0) += matrix[out][out] * a[out]`, which is what
// the generic sum reduces to when every other product is a signed zero (a
// signed zero added to a non-zero or to +0.0 leaves it unchanged, and the
// generic sum never holds -0.0), so it too is bitwise identical. The specialised kernels are
// BITWISE identical to apply_matrix_generic, not merely close: each output amplitude starts from
// zero and adds matrix[out][in] * amplitude[in] in increasing `in` order, as
// the generic loop does, and the target is built with -ffp-contract=off. That
// identity is what keeps CPU and CUDA statevectors, and therefore counts,
// bitwise identical (docs/adr/0014 section 10).
//
// num_threads (>= 1) caps the number of threads that sweep the amplitude groups,
// on MENOS's own pool (src/kernels/cpu/thread_pool.h, docs/adr/0019 section 3).
// Groups are disjoint and each is computed exactly as on one thread, so the
// result is bitwise identical for every num_threads. Gates smaller than
// kParallelMinGroups (below) run on one thread whatever num_threads is.
void apply_matrix(std::span<std::complex<double>> state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands,
                  std::uint32_t num_threads);

// The generic k-operand kernel: gathers each 2^k amplitude group, multiplies
// it by matrix and scatters it back. Same contract as apply_matrix. Kept as the
// oracle the specialised kernels are tested against (docs/adr/0019 section 4);
// apply_matrix calls it for k >= 3.
void apply_matrix_generic(std::span<std::complex<double>> state,
                          std::span<const std::complex<double>> matrix,
                          std::span<const std::uint32_t> operands,
                          std::uint32_t num_threads);

// <psi|P|psi> for the single Pauli term whose X and Z bit masks are x_mask and
// z_mask, by docs/adr/0007's single-pass formula:
//
//   P|m> = i^popcount(x & z) * (-1)^popcount(z & (m ^ x)) * |m ^ x>
//
// so the expectation is i^popcount(x&z) times the sum over m of
// conj(psi[m]) * psi[m ^ x] * (-1)^popcount(z & (m ^ x)).
//
// The coefficient is NOT applied: the caller combines coefficients
// (src/runtime/combine_expectation.h). Complexity: O(2^n).
[[nodiscard]] std::complex<double> expectation_of_term(std::span<const std::complex<double>> state,
                                                       std::uint64_t x_mask,
                                                       std::uint64_t z_mask);

// All Z-only terms (x_mask == 0) at once: p[m] = |psi[m]|^2 followed by one
// in-place Walsh-Hadamard transform gives every <Z-string z> as p[z], so the
// cost is O(n * 2^n) for any number of terms instead of O(terms * 2^n).
// CONSUMES the state: it is overwritten with scratch and must not be read
// afterwards (the dense engine calls this last, after every other output).
// No extra allocation beyond the result, so the engine's peak-bytes estimate
// still holds. Result i has imaginary part +0.0 and real part equal to
// expectation_of_term(state, 0, z_masks[i]) within common::kDefaultAbsTol
// (same products, summed in butterfly instead of index order -- the same
// tolerance the CUDA block-tree reduction is held to, docs/adr/0014 s.14).
// Precondition: state.size() is a power of two; each mask < state.size().
[[nodiscard]] std::vector<std::complex<double>> expectation_of_diagonal_terms(
    std::span<std::complex<double>> state, std::span<const std::uint64_t> z_masks);

}  // namespace cpu

#ifdef MENOS_ENABLE_CUDA

// The same three operations against a device-resident state. Semantics are
// identical to the cpu overloads above, including every convention stated at
// the top of this header; only the memory the state lives in differs.
//
// Each of these throws menos::common::InternalError if the underlying CUDA call
// fails, so a caller never has to inspect a cudaError_t. They synchronize
// before returning, so a subsequent call observes this one's writes.
namespace cuda {

void initialize(DeviceState state, double global_phase);

// matrix and operands are host-side spans; the implementation stages them into
// device memory itself. A gate matrix is 2^k x 2^k, which is tiny next to the
// state, so this is not a transfer worth hoisting.
//
// k = 1 and k = 2 run specialised kernels (docs/adr/0020 section 3) that take
// the matrix as a kernel argument, so they make no host-to-device copy per
// gate; k >= 3 runs apply_matrix_generic. The specialised kernels are BITWISE
// identical to apply_matrix_generic, not merely close: each output amplitude
// starts from zero and adds matrix[out][in] * amplitude[in] in increasing `in`
// order, as the generic kernel does, and the device code is built with
// -fmad=false. That identity is what keeps CPU and CUDA statevectors, and
// therefore counts, bitwise identical (docs/adr/0014 section 10).
void apply_matrix(DeviceState state,
                  std::span<const std::complex<double>> matrix,
                  std::span<const std::uint32_t> operands);

// The generic k-operand kernel: one thread per amplitude group, which gathers
// the 2^k amplitudes, multiplies them by matrix and scatters them back. Same
// contract as apply_matrix. The matrix and operands are copied into
// __constant__ memory before every launch. Kept as the oracle the specialised
// kernels are tested and benchmarked against (docs/adr/0020 section 3);
// apply_matrix calls it for k >= 3.
void apply_matrix_generic(DeviceState state,
                          std::span<const std::complex<double>> matrix,
                          std::span<const std::uint32_t> operands);

// The reduction runs on device and returns a single scalar, so an
// expectation-only job never transfers the 16 * 2^n state back.
//
// NUMERICAL NOTE: the device reduction sums in a different order from the CPU
// loop, so the two agree to floating-point tolerance rather than bitwise. This
// is expected and is not a defect (docs/adr/0014); the summation order here is
// fixed, so the value is reproducible run to run on the same device.
[[nodiscard]] std::complex<double> expectation_of_term(ConstDeviceState state,
                                                       std::uint64_t x_mask,
                                                       std::uint64_t z_mask);

}  // namespace cuda

#endif  // MENOS_ENABLE_CUDA

}  // namespace menos::kernels
