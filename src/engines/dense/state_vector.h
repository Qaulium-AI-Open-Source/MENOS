#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace menos::engines::dense {

// A complex128 statevector of num_qubits qubits. Sized through checked_pow2
// and budget-checked before any allocation: throws ResourceExceeded (naming
// the required and available bytes) rather than letting an allocation throw
// std::bad_alloc.
//
// Stored in a plain std::vector, so a requested statevector is moved into the
// result rather than copied (Phase 21, docs/adr/0025 section 3.3). Until then
// it was 64-byte aligned, which no kernel relied on; results are bitwise
// identical either way.
class StateVector {
public:
    StateVector(int num_qubits, std::uint64_t budget_bytes);

    [[nodiscard]] int num_qubits() const;
    [[nodiscard]] std::uint64_t size_bytes() const;
    [[nodiscard]] std::span<std::complex<double>> amplitudes();
    [[nodiscard]] std::span<const std::complex<double>> amplitudes() const;

    // Hands over the amplitudes, leaving this state empty.
    [[nodiscard]] std::vector<std::complex<double>> release() &&;

private:
    int num_qubits_;
    std::vector<std::complex<double>> data_;
};

// Sets state to exp(i * global_phase) * |0...0>. Precondition: state's
// buffer is freshly constructed (all zero).
void initialize(StateVector& state, double global_phase);

}  // namespace menos::engines::dense
