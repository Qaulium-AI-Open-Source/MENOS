#include "kernels/cuda/device_info.h"

#include <cuda_runtime.h>

#include "kernels/cuda/device_pool.h"

namespace menos::kernels::cuda {
namespace {

DeviceInfo query_device_info() {
    DeviceInfo info;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return info;  // present stays false
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return info;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return info;
    }

    info.present = true;
    info.name = properties.name;
    info.compute_capability_major = properties.major;
    info.compute_capability_minor = properties.minor;
    info.multiprocessor_count = properties.multiProcessorCount;
    info.total_memory_bytes = static_cast<std::uint64_t>(total_bytes);
    info.free_memory_bytes = static_cast<std::uint64_t>(free_bytes);
    return info;
}

}  // namespace

const DeviceInfo& device_info() {
    // Function-local static: queried once, never mutated afterwards. Not the
    // "global mutable state" CLAUDE.md forbids -- written during
    // initialization and read-only thereafter, like the engine registry.
    static const DeviceInfo kInfo = query_device_info();
    return kInfo;
}

std::uint64_t device_free_bytes() {
    if (!device_info().present) {
        return 0;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return static_cast<std::uint64_t>(free_bytes);
}

std::uint64_t device_available_bytes() {
    return device_free_bytes() + device_pool_bytes();
}

std::uint64_t device_pool_bytes() {
    if (!device_info().present) {
        return 0;
    }
    return device_pool().reusable_bytes();
}

}  // namespace menos::kernels::cuda
