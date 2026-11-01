#include "kernels/cuda/device_buffer.h"

#include <cuda_runtime.h>

#include <string>

#include "common/checked_math.h"
#include "common/error.h"
#include "kernels/cuda/cuda_error.h"
#include "kernels/cuda/device_pool.h"

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
    // From the process's pool (Phase 22, docs/adr/0026 section 3.1): the
    // block a previous DeviceBuffer of this size freed, when there is one,
    // instead of a cudaMalloc. Its contents are undefined. initialize()
    // zeroes the state before it writes amplitude 0; until Phase 22 this
    // constructor zeroed it a second time (docs/adr/0026 section 10.1).
    data_ = device_pool().allocate(required_bytes);
    charge_ = common::MemoryLedger::Charge(required_bytes, common::MemoryLedger::Space::Device);
}

DeviceBuffer::~DeviceBuffer() {
    // Back to the pool, which keeps it for the next DeviceBuffer. release()
    // never throws: a destructor cannot, and there is no recovery from a
    // failed free.
    device_pool().release(data_);
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
