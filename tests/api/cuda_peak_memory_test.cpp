// Phase 21 (docs/adr/0025): the device side of the memory report on a real
// device -- Result::peak_device_bytes(), the CUDA candidate's device_bytes and
// the plan's device_available_bytes in explain(), and the free device memory
// read again for every job (section 9, Q3). Skipped (loudly) when no device is
// present.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <format>
#include <string>

#include "engines/dense_cuda/cuda_dense_engine.h"
#include "kernels/cuda/device_buffer.h"
#include "menos/menos.h"

namespace menos {
namespace {

constexpr std::uint64_t kAmplitude = sizeof(std::complex<double>);

Circuit layered(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, 0);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (QubitId q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {q});
            circuit.append(Opcode::T, {q});
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

TEST(CudaPeakMemoryTest, AStatevectorJobHoldsOneStateOnTheDeviceAndOneOnTheHost) {
    if (!engines::dense_cuda::cuda_device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const Result result = run(layered(12), on_cuda());
    ASSERT_EQ(result.plan().regions[0].backend, Backend::Cuda);
    EXPECT_EQ(result.peak_device_bytes(), kAmplitude << 12);
    // The host copy the device state is read back into, moved into the result.
    EXPECT_EQ(result.peak_host_bytes(), kAmplitude << 12);
    EXPECT_LE(result.peak_host_bytes(), result.plan().estimated_state_bytes);
}

TEST(CudaPeakMemoryTest, ExplainShowsTheDeviceBytesAndTheFreeDeviceMemoryItUsed) {
    if (!engines::dense_cuda::cuda_device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const Result result = run(layered(10), on_cuda());
    const Plan& plan = result.plan();
    ASSERT_TRUE(plan.device_available_bytes.has_value());
    const std::string text = result.explain();
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.backend == Backend::Cuda) {
            EXPECT_NE(text.find(std::format(" device_bytes={}", candidate.device.device_bytes)),
                      std::string::npos);
            EXPECT_LE(result.peak_device_bytes(), candidate.device.device_bytes);
        }
    }
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_NE(text.find(std::format(" device_available_bytes={}\n", *plan.device_available_bytes)),
              std::string::npos);
    EXPECT_NE(text.find(std::format(" peak_device_bytes={}", result.peak_device_bytes())),
              std::string::npos);
}

TEST(CudaPeakMemoryTest, AFactorizedPlanWithCudaRegionsEstimatesTheLargestRegionsDeviceState) {
    if (!engines::dense_cuda::cuda_device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    // Two independent blocks of 7 and 5 qubits, both forced onto the device.
    Circuit circuit(12, 0);
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (QubitId q = 0; q < 12; ++q) {
            circuit.append(Opcode::H, {q});
            circuit.append(Opcode::T, {q});
        }
        for (QubitId q = 0; q + 1 < 12; ++q) {
            if (q != 6) {
                circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
            }
        }
    }
    const Result result = run(circuit, on_cuda());
    const Plan& plan = result.plan();
    ASSERT_EQ(plan.strategy, Strategy::Factorized);
    ASSERT_EQ(plan.regions.size(), 2U);
    for (const Region& region : plan.regions) {
        ASSERT_EQ(region.backend, Backend::Cuda);
    }
    const CandidateSummary* winner = nullptr;
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.is_winner) {
            winner = &candidate;
        }
    }
    ASSERT_NE(winner, nullptr);
    // The regions run one after another, so the device holds the 7-qubit
    // state at most; until docs/adr/0025 section 10.5 the estimate was 0.
    EXPECT_GE(winner->device.device_bytes, kAmplitude << 7);
    EXPECT_LT(winner->device.device_bytes, (kAmplitude << 7) + (kAmplitude << 5));
    EXPECT_GT(result.peak_device_bytes(), 0U);
    EXPECT_LE(result.peak_device_bytes(), winner->device.device_bytes);
    EXPECT_NE(result.explain().find(std::format(" device_bytes={}", winner->device.device_bytes)),
              std::string::npos);
}

TEST(CudaPeakMemoryTest, EachJobReadsTheDevicesFreeMemoryAgain) {
    if (!engines::dense_cuda::cuda_device_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const Plan before = preview(layered(10), on_cuda());
    ASSERT_TRUE(before.device_available_bytes.has_value());
    // Held while the next job is planned. Until Phase 21 the figure was read
    // once per process, so this plan would still have been promised it.
    constexpr std::uint64_t kHeldBytes = std::uint64_t{64} << 20;
    const kernels::cuda::DeviceBuffer held(kHeldBytes / kAmplitude,
                                           /*budget_bytes=*/kHeldBytes);
    const Plan after = preview(layered(10), on_cuda());
    ASSERT_TRUE(after.device_available_bytes.has_value());
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_GE(*before.device_available_bytes, *after.device_available_bytes + kHeldBytes)
        << "before " << *before.device_available_bytes << ", after "
        << *after.device_available_bytes;
    // NOLINTEND(bugprone-unchecked-optional-access)
}

}  // namespace
}  // namespace menos
