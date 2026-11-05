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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/error.h"
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

// Phase 23 (docs/adr/0027 section 9, Q8(i)): memory another process took
// between the plan and the allocation is a named resource refusal, not an
// internal error. A direct driver allocation outside the pool stands in for
// the other process. The states are then asked for one after another until
// the driver refuses: on a native Linux driver the first request is refused
// (32 MiB left, 256 MiB asked), while WSL's driver backs device memory with
// shared system memory and grants allocations past the free figure, by an
// amount that varies between calls (measured up to 16 GiB, docs/adr/0027
// section 10.3), so the refusal comes later there. Either way it must come,
// and it must be the named refusal.
TEST(CudaDevicePoolTest, ADeviceThatRanOutSinceThePlanIsARefusalNamingTheBytes) {
    if (!device_info().present) {
        GTEST_SKIP() << "no CUDA device";
    }
    device_pool().trim();
    constexpr std::uint64_t kHeadroom = std::uint64_t{32} << 20;
    constexpr std::size_t kStateAmplitudes = std::size_t{1} << 24;  // 256 MiB
    constexpr std::size_t kMostStates = 256;  // 64 GiB: more than any shared-memory grant
    const std::uint64_t free_before = device_free_bytes();
    if (free_before <= 2 * kHeadroom) {
        GTEST_SKIP() << "the device has too little free memory to stage the race";
    }
    DevicePool other_process(/*use_pool=*/false);
    void* hog = other_process.allocate(free_before - kHeadroom);
    ASSERT_NE(hog, nullptr);
    {
        std::vector<std::unique_ptr<DeviceBuffer>> states;
        bool refused = false;
        while (!refused && states.size() < kMostStates) {
            try {
                // A state the plan would have found room for, asked for
                // after the device filled up.
                states.push_back(std::make_unique<DeviceBuffer>(
                    kStateAmplitudes, std::numeric_limits<std::uint64_t>::max()));
            } catch (const common::ResourceExceeded& e) {
                refused = true;
                const std::string what = e.what();
                EXPECT_NE(what.find("could not provide " +
                                    std::to_string(std::uint64_t{16} * kStateAmplitudes) +
                                    " bytes of device memory"),
                          std::string::npos)
                    << what;
                EXPECT_NE(what.find("bytes are free now"), std::string::npos) << what;
                EXPECT_NE(what.find("retained in MENOS's pool"), std::string::npos) << what;
            }
        }
        EXPECT_TRUE(refused) << "the driver granted " << states.size()
                             << " states of 256 MiB past the free figure without refusing";
    }
    device_pool().trim();
    other_process.release(hog);
    // The device works again once the memory is back, and the sticky error
    // was cleared.
    DeviceBuffer after(kBlockAmplitudes, std::numeric_limits<std::uint64_t>::max());
    initialize(after.state(), 0.0);
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
