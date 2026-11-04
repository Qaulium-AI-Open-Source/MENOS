#pragma once

#include <cstdint>

namespace menos::kernels::cuda {

// The device memory MENOS allocates from (Phase 22, docs/adr/0026 section
// 3.1): one explicit CUDA stream-ordered memory pool, which keeps the memory
// a DeviceBuffer frees and serves the next DeviceBuffer from it, instead of a
// cudaMalloc and a cudaFree per execution. On this GPU a direct allocate,
// zero-fill and free cycle costs 1-3 ms at 20 qubits and 9-23 ms at 24,
// against 1-6 us from the pool (docs/adr/0026 section 10.1).
//
// The pool's release threshold is the device's total memory (owner, Q2), so
// nothing is returned to the driver on its own: the memory the pool retains
// between jobs is reported by reusable_bytes(), counted as available by the
// planner (device_info.h, device_available_bytes()), printed by explain(),
// and released by menos::release_device_memory() (trim()) or at process
// exit. Allocation and release are stream-ordered on the legacy default
// stream, the stream every MENOS kernel runs on, and allocate() waits for
// the allocation before returning, so a pointer it hands out is usable by
// the next host call exactly as a cudaMalloc'ed one is.
//
// Memory from the pool is NOT zero-filled: it holds whatever the previous
// user left. kernels::cuda::initialize() zeroes the whole state before it
// writes amplitude 0, and nothing in MENOS reads a buffer before that.
//
// A device that reports no memory-pool support (cudaDevAttrMemoryPoolsSupported
// = 0), or on which the pool cannot be created, falls back to plain
// cudaMalloc/cudaFree with pooled() == false; reusable_bytes() is then 0 and
// explain() says so. Like device_info(), the process-wide pool is created on
// first use and never mutated afterwards except through the CUDA runtime's
// own thread-safe calls.
class DevicePool {
public:
    // use_pool = false takes the fallback path unconditionally. Production
    // code constructs only device_pool() below, with true; the flag exists so
    // the fallback has a test on a machine whose device supports pools.
    explicit DevicePool(bool use_pool);
    ~DevicePool();

    DevicePool(const DevicePool&) = delete;
    DevicePool& operator=(const DevicePool&) = delete;
    DevicePool(DevicePool&&) = delete;
    DevicePool& operator=(DevicePool&&) = delete;

    // Whether allocations come from the pool (true) or from cudaMalloc.
    [[nodiscard]] bool pooled() const;

    // `bytes` of device memory, contents undefined. Callers check their
    // budget before asking (DeviceBuffer); when the device still cannot
    // provide the bytes -- another process took device memory between the
    // plan and the allocation (Phase 23, docs/adr/0027 section 9, Q8(i)) --
    // throws ResourceExceeded naming the bytes asked for, the bytes free now
    // and the bytes the pool retains. Any other CUDA failure is an
    // InternalError.
    [[nodiscard]] void* allocate(std::uint64_t bytes);

    // Returns `data` (from allocate(), or nullptr) to the pool or the driver.
    // Never throws: it runs from destructors.
    void release(void* data) noexcept;

    // The bytes the pool holds reserved but not in use: what the next
    // allocation can take without asking the driver. 0 when not pooled.
    [[nodiscard]] std::uint64_t reusable_bytes() const;

    // Returns every reusable byte to the driver. Nothing when not pooled.
    void trim();

private:
    // A cudaMemPool_t, kept untyped so this header names no CUDA type (as
    // device_buffer.h and dense_kernels.h do not).
    void* pool_ = nullptr;
    bool pooled_ = false;
};

// The one pool every DeviceBuffer allocates from, created on first use.
[[nodiscard]] DevicePool& device_pool();

}  // namespace menos::kernels::cuda
