#include "engines/dense/state_vector.h"

#include <utility>

#include "common/checked_math.h"
#include "common/error.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense {

// num_qubits (a small qubit count) and budget_bytes (a byte count, always
// vastly larger) are not realistically confusable in a call site; the
// declaration order matches every other constructor in this codebase
// (size, then budget).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
StateVector::StateVector(int num_qubits, std::uint64_t budget_bytes) : num_qubits_(num_qubits) {
    common::ensure(num_qubits >= 0, "StateVector: num_qubits must be non-negative");
    const std::uint64_t num_amplitudes = common::checked_pow2(static_cast<unsigned>(num_qubits));
    const std::uint64_t required_bytes =
        common::checked_mul(num_amplitudes, sizeof(std::complex<double>));
    if (required_bytes > budget_bytes) {
        throw common::ResourceExceeded("StateVector: required " + std::to_string(required_bytes) +
                                       " bytes exceeds the budget of " +
                                       std::to_string(budget_bytes) + " bytes");
    }
    data_.resize(num_amplitudes);
    charge_.set(common::buffer_bytes(data_));
}

int StateVector::num_qubits() const {
    return num_qubits_;
}

std::uint64_t StateVector::size_bytes() const {
    return static_cast<std::uint64_t>(data_.size()) * sizeof(std::complex<double>);
}

std::span<std::complex<double>> StateVector::amplitudes() {
    return {data_.data(), data_.size()};
}

std::span<const std::complex<double>> StateVector::amplitudes() const {
    return {data_.data(), data_.size()};
}

std::vector<std::complex<double>> StateVector::release() && {
    // From here the buffer is the result's, charged by whoever holds it.
    charge_.release();
    return std::move(data_);
}

void initialize(StateVector& state, double global_phase) {
    kernels::cpu::initialize(state.amplitudes(), global_phase);
}

}  // namespace menos::engines::dense
