#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

#include "common/memory_ledger.h"
#include "kernels/dense_kernels.h"

namespace menos::kernels::cuda {

// An owning device allocation of complex128 amplitudes (docs/adr/0014).
//
// The device-memory analogue of engines::dense::StateVector, and deliberately
// the same shape: the requirement is checked against the budget BEFORE any
// allocation is attempted, and exceeding it throws ResourceExceeded naming the
// required and available byte counts -- never a raw cudaMalloc failure, exactly
// as StateVector refuses to let std::bad_alloc escape (CLAUDE.md: "The budget
// is checked before any allocation, and failure throws ResourceExceeded").
//
// Device memory is a budget SEPARATE from host memory. This GPU reports far
// less usable memory than its nameplate (3770 MiB of a nominal 4096 MiB), so
// the budget passed here comes from cudaMemGetInfo, never from a datasheet.
class DeviceBuffer {
public:
    // Allocates num_amplitudes * 16 bytes on the current device, from the
    // process's memory pool (device_pool.h, Phase 22). The contents are
    // UNDEFINED until kernels::cuda::initialize() runs; nothing reads a state
    // before that. Throws ResourceExceeded if the size exceeds budget_bytes;
    // throws InternalError if the allocation itself fails despite fitting the
    // budget.
    DeviceBuffer(std::size_t num_amplitudes, std::uint64_t budget_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    [[nodiscard]] DeviceState state();
    [[nodiscard]] ConstDeviceState state() const;
    [[nodiscard]] std::size_t num_amplitudes() const;
    [[nodiscard]] std::uint64_t size_bytes() const;

    // Copies the whole buffer to host. This is the 16 * 2^n device-to-host
    // transfer the cost model accounts for; an expectation-only job never
    // calls it.
    void copy_to_host(std::span<std::complex<double>> destination) const;

private:
    void* data_;
    std::size_t num_amplitudes_;
    common::MemoryLedger::Charge charge_;
};

}  // namespace menos::kernels::cuda
