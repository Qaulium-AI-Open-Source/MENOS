// The device memory pool (Phase 22, docs/adr/0026 section 3.1): memory a
// DeviceBuffer frees stays in the pool and serves the next allocation; the
// retained bytes are reported, counted as available and returned on trim();
// and the fallback path, for a device without memory-pool support, allocates
// and frees through cudaMalloc/cudaFree with nothing retained.
//
// Skipped (loudly) when no CUDA device is present; it never passes without
// one.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <vector>

#include "kernels/cuda/device_buffer.h"
#include "kernels/cuda/device_info.h"
#include "kernels/cuda/device_pool.h"
#include "kernels/dense_kernels.h"

namespace menos::kernels::cuda {
namespace {

constexpr std::uint64_t kBlockBytes = std::uint64_t{64} << 20;  // 64 MiB
constexpr std::size_t kBlockAmplitudes = kBlockBytes / sizeof(std::complex<double>);

TEST(CudaDevicePoolTest, TheProcessPoolIsInUseWhenTheDeviceSupportsPools) {
    if (!device_info().present) {
        GTEST_SKIP() << "no CUDA device";
    }
    // This machine's device supports pools (docs/adr/0026 section 10.1). A
    // device that does not takes the fallback, which the test below covers.
    EXPECT_TRUE(device_pool().pooled()) << "the device reports no memory pool support";
}

TEST(CudaDevicePoolTest, AFreedBlockIsRetainedAndServesTheNextAllocation) {
    if (!device_info().present) {
        GTEST_SKIP() << "no CUDA device";
    }
    DevicePool pool(/*use_pool=*/true);
    if (!pool.pooled()) {
        GTEST_SKIP() << "the device reports no memory pool support";
    }
    EXPECT_EQ(pool.reusable_bytes(), 0U);
    void* first = pool.allocate(kBlockBytes);
    ASSERT_NE(first, nullptr);
    // While the block is in use nothing is reusable (the pool may round its
    // reservation up, so "nothing" is "less than the block").
    EXPECT_LT(pool.reusable_bytes(), kBlockBytes);
    pool.release(first);
    const std::uint64_t retained = pool.reusable_bytes();
    EXPECT_GE(retained, kBlockBytes);

    // The second allocation comes out of the retained bytes, not the driver:
    // the pool's reusable figure falls by the block and the device's free
    // memory does not fall by it.
    const std::uint64_t free_before = device_free_bytes();
    void* second = pool.allocate(kBlockBytes);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(pool.reusable_bytes(), retained - kBlockBytes);
    EXPECT_GT(device_free_bytes() + kBlockBytes / 2, free_before)
        << "free before " << free_before << ", after " << device_free_bytes();
    pool.release(second);
    pool.trim();
    EXPECT_EQ(pool.reusable_bytes(), 0U);
}

TEST(CudaDevicePoolTest, TheFallbackAllocatesFromTheDriverAndRetainsNothing) {
    if (!device_info().present) {
        GTEST_SKIP() << "no CUDA device";
    }
    DevicePool pool(/*use_pool=*/false);
    EXPECT_FALSE(pool.pooled());
    void* data = pool.allocate(kBlockBytes);
    ASSERT_NE(data, nullptr);
    // Usable as a state: initialize() writes every byte of it.
    initialize(DeviceState{data, kBlockAmplitudes}, 0.0);
    EXPECT_EQ(pool.reusable_bytes(), 0U);
    pool.release(data);
    EXPECT_EQ(pool.reusable_bytes(), 0U);
    pool.trim();  // nothing to do, and must not throw
}

TEST(CudaDevicePoolTest, AvailableBytesCountTheRetainedPool) {
    if (!device_info().present) {
        GTEST_SKIP() << "no CUDA device";
    }
    if (!device_pool().pooled()) {
        GTEST_SKIP() << "the device reports no memory pool support";
    }
    {
        const DeviceBuffer buffer(kBlockAmplitudes, device_available_bytes());
        static_cast<void>(buffer);
    }
    // The buffer's block is back in the pool: reported, and counted as
    // available on top of the device's free memory (docs/adr/0026 section
    // 3.4), so the next job of this size is not refused for lack of memory
    // that is waiting for it.
    EXPECT_GE(device_pool_bytes(), kBlockBytes);
    EXPECT_GE(device_available_bytes(), device_free_bytes() + kBlockBytes);
    device_pool().trim();
    EXPECT_EQ(device_pool_bytes(), 0U);
}

TEST(CudaDevicePoolTest, APooledStateIsInitializedExactlyAsBefore) {
    if (!device_info().present) {
        GTEST_SKIP() << "no CUDA device";
    }
    // A block that comes back from the pool holds the previous user's bytes;
    // initialize() must leave the state |0...0> regardless.
    constexpr std::size_t kAmplitudes = std::size_t{1} << 12;
    std::vector<std::complex<double>> host(kAmplitudes);
    {
        DeviceBuffer dirty(kAmplitudes, device_available_bytes());
        initialize(dirty.state(), 0.0);
        const std::vector<std::complex<double>> hadamard{{0.70710678118654752, 0.0},
                                                         {0.70710678118654752, 0.0},
                                                         {0.70710678118654752, 0.0},
                                                         {-0.70710678118654752, 0.0}};
        for (std::uint32_t q = 0; q < 12; ++q) {
            apply_matrix(dirty.state(), hadamard, std::vector<std::uint32_t>{q});
        }
    }
    DeviceBuffer fresh(kAmplitudes, device_available_bytes());
    initialize(fresh.state(), 0.25);
    fresh.copy_to_host(host);
    const std::complex<double> phase = std::exp(std::complex<double>{0.0, 0.25});
    EXPECT_EQ(host[0], phase);
    for (std::size_t i = 1; i < kAmplitudes; ++i) {
        ASSERT_EQ(host[i], (std::complex<double>{0.0, 0.0})) << "amplitude " << i;
    }
}

}  // namespace
}  // namespace menos::kernels::cuda
