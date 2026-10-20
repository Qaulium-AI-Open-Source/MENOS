#include "kernels/cuda/device_buffer.h"

#include <cuda_runtime.h>

#include <string>

#include "common/checked_math.h"
#include "common/error.h"
#include "kernels/cuda/cuda_error.h"

namespace menos::kernels::cuda {

// num_amplitudes (a count) and budget_bytes (a byte count) are not realistically
// confusable at a call site, and the order matches StateVector's (size, then
// budget).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
DeviceBuffer::DeviceBuffer(std::size_t num_amplitudes, std::uint64_t budget_bytes)
    : data_(nullptr), num_amplitudes_(num_amplitudes) {
    const std::uint64_t required_bytes = common::checked_mul(
        static_cast<std::uint64_t>(num_amplitudes), sizeof(std::complex<double>));
    if (required_bytes > budget_bytes) {
        throw common::ResourceExceeded("DeviceBuffer: required " + std::to_string(required_bytes) +
                                       " bytes of device memory exceeds the device budget of " +
                                       std::to_string(budget_bytes) + " bytes");
    }
    ensure_cuda(cudaMalloc(&data_, required_bytes), "cudaMalloc");
    charge_ = common::MemoryLedger::Charge(required_bytes, common::MemoryLedger::Space::Device);
    // Zero-filled so initialize() only has to write element 0. cudaMemset is
    // used rather than a kernel because a byte-wise zero fill is exactly an
    // all-zero complex128 buffer.
    ensure_cuda(cudaMemset(data_, 0, required_bytes), "cudaMemset");
}

DeviceBuffer::~DeviceBuffer() {
    // A destructor cannot throw, and there is no recovery from a failed free.
    // The value is deliberately discarded rather than checked.
    static_cast<void>(cudaFree(data_));
}

DeviceState DeviceBuffer::state() {
    return DeviceState{data_, num_amplitudes_};
}

ConstDeviceState DeviceBuffer::state() const {
    return ConstDeviceState{data_, num_amplitudes_};
}

std::size_t DeviceBuffer::num_amplitudes() const {
    return num_amplitudes_;
}

std::uint64_t DeviceBuffer::size_bytes() const {
    return static_cast<std::uint64_t>(num_amplitudes_) * sizeof(std::complex<double>);
}

void DeviceBuffer::copy_to_host(std::span<std::complex<double>> destination) const {
    common::ensure(destination.size() == num_amplitudes_,
                   "DeviceBuffer::copy_to_host: destination size does not match the buffer");
    ensure_cuda(cudaMemcpy(destination.data(), data_, size_bytes(), cudaMemcpyDeviceToHost),
                "cudaMemcpy device-to-host");
}

}  // namespace menos::kernels::cuda
