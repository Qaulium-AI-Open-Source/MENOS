// Phase 22 (docs/adr/0026): the device memory pool seen from a job. A second
// job of the same size is served from the memory the first one freed; the
// pool's retained bytes are counted in the plan's device availability and
// printed by explain(); menos::release_device_memory() empties the pool; and
// the memory ledger reports the same peak as before, pooled or not. Skipped
// (loudly) when no device is present.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <format>
#include <string>

#include "engines/dense_cuda/cuda_dense_engine.h"
#include "kernels/cuda/device_info.h"
#include "kernels/cuda/device_pool.h"
#include "menos/menos.h"

namespace menos {
namespace {

constexpr std::uint64_t kAmplitude = sizeof(std::complex<double>);
constexpr std::uint32_t kQubits = 20;  // a 16 MiB state
constexpr std::uint64_t kStateBytes = kAmplitude << kQubits;

Circuit layered(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, 0);
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        for (QubitId q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {q});
        }
        for (QubitId q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

RunOptions on_cuda() {
    RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
    options.backend = Backend::Cuda;
    return options;
}

bool skip_without_pool() {
    if (!engines::dense_cuda::cuda_device_available()) {
        return true;
    }
    return !kernels::cuda::device_pool().pooled();
}

TEST(CudaMemoryPoolTest, ASecondJobOfTheSameSizeIsServedFromThePool) {
    if (skip_without_pool()) {
        GTEST_SKIP() << "no CUDA device, or no memory pool support";
    }
    release_device_memory();
    const Result first = run(layered(kQubits), on_cuda());
    ASSERT_EQ(first.plan().regions[0].backend, Backend::Cuda);
    EXPECT_EQ(first.peak_device_bytes(), kStateBytes);
    // The first job's state is back in the pool.
    EXPECT_GE(kernels::cuda::device_pool_bytes(), kStateBytes);

    const std::uint64_t free_before = kernels::cuda::device_free_bytes();
    const Result second = run(layered(kQubits), on_cuda());
    // Served from the pool: the device's free memory did not fall by a state,
    // and the ledger reports the state exactly as before.
    EXPECT_GT(kernels::cuda::device_free_bytes() + kStateBytes / 2, free_before)
        << "free before " << free_before << ", after " << kernels::cuda::device_free_bytes();
    EXPECT_EQ(second.peak_device_bytes(), kStateBytes);
    EXPECT_EQ(first.statevector(), second.statevector());
}

TEST(CudaMemoryPoolTest, ThePlanCountsAndExplainPrintsTheRetainedPool) {
    if (skip_without_pool()) {
        GTEST_SKIP() << "no CUDA device, or no memory pool support";
    }
    static_cast<void>(run(layered(kQubits), on_cuda()));
    const Result result = run(layered(kQubits), on_cuda());
    const Plan& plan = result.plan();
    ASSERT_TRUE(plan.device_available_bytes.has_value());
    ASSERT_TRUE(plan.device_pool_bytes.has_value());
    ASSERT_TRUE(plan.device_pooled.has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_TRUE(*plan.device_pooled);
    EXPECT_GE(*plan.device_pool_bytes, kStateBytes);
    EXPECT_GE(*plan.device_available_bytes, *plan.device_pool_bytes);
    const std::string text = result.explain();
    EXPECT_NE(text.find(std::format(" device_available_bytes={} device_pool_bytes={} pooled=true\n",
                                    *plan.device_available_bytes,
                                    *plan.device_pool_bytes)),
              std::string::npos)
        << text;
    // NOLINTEND(bugprone-unchecked-optional-access)
}

TEST(CudaMemoryPoolTest, ReleaseDeviceMemoryEmptiesThePool) {
    if (skip_without_pool()) {
        GTEST_SKIP() << "no CUDA device, or no memory pool support";
    }
    static_cast<void>(run(layered(kQubits), on_cuda()));
    ASSERT_GE(kernels::cuda::device_pool_bytes(), kStateBytes);
    const std::uint64_t free_before = kernels::cuda::device_free_bytes();
    release_device_memory();
    EXPECT_EQ(kernels::cuda::device_pool_bytes(), 0U);
    EXPECT_GE(kernels::cuda::device_free_bytes() + kStateBytes / 2, free_before + kStateBytes)
        << "free before " << free_before << ", after " << kernels::cuda::device_free_bytes();
    // Idempotent.
    release_device_memory();
    EXPECT_EQ(kernels::cuda::device_pool_bytes(), 0U);
}

TEST(CudaMemoryPoolTest, APlanThatNeedsExactlyTheRetainedMemoryIsFeasible) {
    if (skip_without_pool()) {
        GTEST_SKIP() << "no CUDA device, or no memory pool support";
    }
    // The planner checks a CUDA candidate against device_available_bytes,
    // which counts the retained pool (docs/adr/0026 section 3.4). The two
    // previews must agree on availability up to the retained block: without
    // the pool term, the second preview would see a state less.
    release_device_memory();
    const Plan before = preview(layered(kQubits), on_cuda());
    static_cast<void>(run(layered(kQubits), on_cuda()));
    const Plan after = preview(layered(kQubits), on_cuda());
    ASSERT_TRUE(before.device_available_bytes.has_value());
    ASSERT_TRUE(after.device_available_bytes.has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_GT(*after.device_available_bytes + kStateBytes / 2, *before.device_available_bytes)
        << "before " << *before.device_available_bytes << ", after "
        << *after.device_available_bytes;
    // NOLINTEND(bugprone-unchecked-optional-access)
}

}  // namespace
}  // namespace menos
