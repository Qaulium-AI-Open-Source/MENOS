#include "kernels/cuda/device_pool.h"

#include <cuda_runtime.h>

#include <string>

#include "common/error.h"
#include "kernels/cuda/cuda_error.h"
#include "kernels/cuda/device_info.h"

namespace menos::kernels::cuda {
namespace {

// The CUDA runtime documents the pool attributes as cuuint64_t, which is
// unsigned long long; cuda_runtime.h alone does not declare the typedef.
using PoolAttributeValue = unsigned long long;  // NOLINT(google-runtime-int)

cudaMemPool_t as_pool(void* pool) {
    return static_cast<cudaMemPool_t>(pool);
}

}  // namespace

DevicePool::DevicePool(bool use_pool) {
    if (!use_pool || !device_info().present) {
        return;
    }
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0) != cudaSuccess ||
        supported == 0) {
        return;
    }
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaMemPoolCreate(&pool, &props) != cudaSuccess) {
        return;
    }
    // Owner decision Q2 (docs/adr/0026 section 9): the pool may hold up to the
    // device's total memory, so freed memory never goes back to the driver on
    // its own. The CUDA default is 0, which would release everything at the
    // next synchronisation, and MENOS synchronises after every gate.
    PoolAttributeValue threshold = device_info().total_memory_bytes;
    if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) != cudaSuccess) {
        static_cast<void>(cudaMemPoolDestroy(pool));
        return;
    }
    pool_ = pool;
    pooled_ = true;
}

DevicePool::~DevicePool() {
    // The process-wide pool is destroyed after main() returns, when the CUDA
    // runtime may already be unloading; a destructor cannot throw and there is
    // nothing to do about a failed destroy, so the status is discarded.
    if (pool_ != nullptr) {
        static_cast<void>(cudaMemPoolDestroy(as_pool(pool_)));
    }
}

bool DevicePool::pooled() const {
    return pooled_;
}

void* DevicePool::allocate(std::uint64_t bytes) {
    void* data = nullptr;
    const cudaError_t status = pooled_
                                   ? cudaMallocFromPoolAsync(&data, bytes, as_pool(pool_), nullptr)
                                   : cudaMalloc(&data, bytes);
    if (status == cudaErrorMemoryAllocation) {
        // The device ran out between the plan and the allocation: another
        // process took the memory the plan was checked against (Phase 23,
        // docs/adr/0027 section 9, Q8(i)). A resource refusal, named, not an
        // internal error: nothing in MENOS is wrong. The sticky error is
        // cleared so the next call starts clean.
        static_cast<void>(cudaGetLastError());
        throw common::ResourceExceeded(
            "the CUDA device could not provide " + std::to_string(bytes) +
            " bytes of device memory: " + std::to_string(device_free_bytes()) +
            " bytes are free now and " + std::to_string(reusable_bytes()) +
            " are retained in MENOS's pool; another process may have taken device memory since "
            "the job was planned");
    }
    ensure_cuda(status, pooled_ ? "cudaMallocFromPoolAsync" : "cudaMalloc");
    if (pooled_) {
        ensure_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize (allocation)");
    }
    return data;
}

void DevicePool::release(void* data) noexcept {
    if (data == nullptr) {
        return;
    }
    if (pooled_) {
        static_cast<void>(cudaFreeAsync(data, nullptr));
    } else {
        static_cast<void>(cudaFree(data));
    }
}

std::uint64_t DevicePool::reusable_bytes() const {
    if (!pooled_) {
        return 0;
    }
    // A release() is stream-ordered and may not have landed yet, in which
    // case the pool still counts its block as used. The figure is read at
    // job boundaries (plan_job, explain, tests), where nothing is in flight,
    // so waiting costs nothing and makes the figure exact.
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
        return 0;
    }
    PoolAttributeValue reserved = 0;
    PoolAttributeValue used = 0;
    if (cudaMemPoolGetAttribute(as_pool(pool_), cudaMemPoolAttrReservedMemCurrent, &reserved) !=
            cudaSuccess ||
        cudaMemPoolGetAttribute(as_pool(pool_), cudaMemPoolAttrUsedMemCurrent, &used) !=
            cudaSuccess) {
        return 0;
    }
    return reserved > used ? static_cast<std::uint64_t>(reserved - used) : 0;
}

void DevicePool::trim() {
    if (!pooled_) {
        return;
    }
    // Pending frees first (see reusable_bytes), or they would survive the trim.
    ensure_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize (trim)");
    ensure_cuda(cudaMemPoolTrimTo(as_pool(pool_), 0), "cudaMemPoolTrimTo");
}

DevicePool& device_pool() {
    // Function-local static, like device_info(): created on first use and not
    // mutated afterwards except through the runtime's own thread-safe calls.
    static DevicePool pool(/*use_pool=*/true);
    return pool;
}

}  // namespace menos::kernels::cuda
