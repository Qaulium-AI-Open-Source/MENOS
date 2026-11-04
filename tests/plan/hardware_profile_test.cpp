#include "plan/hardware_profile.h"

#include <gtest/gtest.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace menos::plan {
namespace {

TEST(DetectHardwareProfileTest, ReturnsAPositiveMemorySize) {
    const HardwareProfile hw = detect_hardware_profile();
    EXPECT_GT(hw.total_memory_bytes, 0U);
}

#ifdef _WIN32
// The Windows probe (GlobalMemoryStatusEx) checked against an independent
// source: the RAM installed in the machine, as the firmware reports it. The
// memory the OS can use is a subset of what is installed, so it is positive
// and never larger (docs/adr/0022 section 2).
TEST(DetectHardwareProfileTest, WindowsTotalIsBoundedByInstalledMemory) {
    ULONGLONG installed_kib = 0;
    ASSERT_NE(GetPhysicallyInstalledSystemMemory(&installed_kib), 0);
    const std::uint64_t installed_bytes = static_cast<std::uint64_t>(installed_kib) * 1024U;
    const HardwareProfile hw = detect_hardware_profile();
    EXPECT_GT(hw.total_memory_bytes, 0U);
    EXPECT_LE(hw.total_memory_bytes, installed_bytes);
}
#endif

// detect_ is the uncalibrated path, and the planner distinguishes the two by
// exactly this: a zero in either figure means "never calibrated" and makes a
// time-budgeted job an error rather than a silent pass.
TEST(DetectHardwareProfileTest, LeavesBothCalibratedFiguresAtZero) {
    const HardwareProfile hw = detect_hardware_profile();
    EXPECT_EQ(hw.memory_bandwidth_bytes_per_second, 0.0);
    EXPECT_EQ(hw.work_units_per_second, 0.0);
}

TEST(CalibrateHardwareTest, ProducesPositiveBandwidthAndThroughput) {
    const HardwareProfile hw = calibrate_hardware();
    EXPECT_GT(hw.total_memory_bytes, 0U);
    EXPECT_GT(hw.memory_bandwidth_bytes_per_second, 0.0);
    EXPECT_GT(hw.work_units_per_second, 0.0);
}

// A regression test for a real defect, not a smoke test. With the volatile
// observer removed from measure_memory_bandwidth()'s loop, GCC at -O3 deleted
// every memcpy as a dead store and the warm-up pass reported 558691 copies in
// 15 ms -- 2.5 PB/s. Any figure above this ceiling means the copies are being
// elided again and the measurement is timing an empty loop.
//
// The bound is deliberately far above any real machine (1 TB/s; the fastest
// current server memory subsystems are an order of magnitude below it) so the
// test cannot fail on fast hardware. It bounds "did the work happen at all",
// which is the thing that actually broke.
TEST(CalibrateHardwareTest, BandwidthIsPhysicallyPlausible) {
    const HardwareProfile hw = calibrate_hardware();
    constexpr double kImplausibleBytesPerSecond = 1.0e12;
    EXPECT_LT(hw.memory_bandwidth_bytes_per_second, kImplausibleBytesPerSecond)
        << "measured " << hw.memory_bandwidth_bytes_per_second
        << " B/s, which exceeds any real memory subsystem -- the timed copies "
           "are almost certainly being optimized away";
}

// The figures are measurements, so they are not expected to be bit-identical
// between calls. They are expected to be stable enough to plan against: if
// repeated calibration on an idle machine disagreed by an order of magnitude,
// a time budget would admit a candidate on one run and reject it on the next.
//
// Bandwidth is a property of the hardware and is the tighter of the two.
// Observed spread across builds during development was ~12%; the 4x band here
// is loose enough to survive a loaded CI machine while still catching the
// failure mode that matters (a figure that is not reproducible at all).
TEST(CalibrateHardwareTest, RepeatedCalibrationAgreesWithinAnOrderOfMagnitude) {
    std::vector<double> bandwidths;
    std::vector<double> throughputs;
    for (int i = 0; i < 3; ++i) {
        const HardwareProfile hw = calibrate_hardware();
        bandwidths.push_back(hw.memory_bandwidth_bytes_per_second);
        throughputs.push_back(hw.work_units_per_second);
    }

    const auto spread = [](const std::vector<double>& values) {
        const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
        return *hi / *lo;
    };
    EXPECT_LT(spread(bandwidths), 4.0) << "bandwidth is not reproducible";
    EXPECT_LT(spread(throughputs), 4.0) << "throughput is not reproducible";
}

// The cache is what makes calibration affordable: without it every job in a
// process would repeat the microbenchmarks.
TEST(CalibratedHardwareProfileTest, IsCachedSoRepeatedCallsReturnTheSameObject) {
    const HardwareProfile& first = calibrated_hardware_profile();
    const HardwareProfile& second = calibrated_hardware_profile();
    EXPECT_EQ(&first, &second);
    EXPECT_EQ(first.memory_bandwidth_bytes_per_second, second.memory_bandwidth_bytes_per_second);
    EXPECT_EQ(first.work_units_per_second, second.work_units_per_second);
}

TEST(CalibratedHardwareProfileTest, CarriesTheCalibratedFigures) {
    const HardwareProfile& hw = calibrated_hardware_profile();
    EXPECT_GT(hw.total_memory_bytes, 0U);
    EXPECT_GT(hw.memory_bandwidth_bytes_per_second, 0.0);
    EXPECT_GT(hw.work_units_per_second, 0.0);
}

// ---- The scratch directory (Phase 23, docs/adr/0027 section 3.3) ---------

std::size_t scratch_files_in(const std::filesystem::path& directory) {
#if defined(_WIN32)
    const auto pid = static_cast<unsigned long>(_getpid());
#else
    const auto pid = static_cast<unsigned long>(getpid());
#endif
    // This process's files only: other test processes share the directory.
    const std::string prefix = "menos-scratch-" + std::to_string(pid) + "-";
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().filename().string().starts_with(prefix)) {
            ++count;
        }
    }
    return count;
}

TEST(ScratchProfileTest, TheDirectorysSpaceIsReadAndItsBandwidthOnlyOnRequest) {
    const std::filesystem::path directory = std::filesystem::current_path();
    const HardwareProfile unmeasured =
        current_hardware_profile(directory, /*measure_scratch=*/false);
    EXPECT_EQ(unmeasured.scratch_directory, directory);
    EXPECT_GT(unmeasured.scratch_capacity_bytes, 0U);
    EXPECT_GT(unmeasured.scratch_available_bytes, 0U);
    EXPECT_LE(unmeasured.scratch_available_bytes, unmeasured.scratch_capacity_bytes);
    EXPECT_EQ(unmeasured.scratch_write_bytes_per_second, 0.0);
    EXPECT_EQ(unmeasured.scratch_read_bytes_per_second, 0.0);
    // Everything else is the calibrated profile.
    EXPECT_EQ(unmeasured.total_memory_bytes, calibrated_hardware_profile().total_memory_bytes);
    EXPECT_EQ(unmeasured.work_units_per_second,
              calibrated_hardware_profile().work_units_per_second);

    // A directory that does not exist leaves every scratch figure zero.
    const HardwareProfile missing =
        current_hardware_profile(directory / "no-such-directory-2311", /*measure_scratch=*/true);
    EXPECT_TRUE(missing.scratch_directory.empty());
    EXPECT_EQ(missing.scratch_capacity_bytes, 0U);
    EXPECT_EQ(missing.scratch_available_bytes, 0U);
}

TEST(ScratchProfileTest, TheMeasurementWritesWholeChunksAndRemovesItsFile) {
    const std::filesystem::path directory = std::filesystem::current_path();
    const std::size_t before = scratch_files_in(directory);
    // 64 MiB asked: rounded up to one chunk of 256 MiB, the smallest the
    // tier prefers.
    const ScratchBandwidth small = measure_scratch_bandwidth(directory, std::uint64_t{64} << 20);
    EXPECT_EQ(small.measured_bytes, std::uint64_t{256} << 20);
    EXPECT_GT(small.write_bytes_per_second, 0.0);
    EXPECT_GT(small.read_bytes_per_second, 0.0);
    EXPECT_EQ(scratch_files_in(directory), before);
    // Never more than 1 GiB, whatever is asked.
    const ScratchBandwidth capped = measure_scratch_bandwidth(directory, std::uint64_t{1} << 40);
    EXPECT_EQ(capped.measured_bytes, std::uint64_t{1} << 30);
    EXPECT_EQ(scratch_files_in(directory), before);
    // A directory that cannot be written measures nothing.
    const ScratchBandwidth none =
        measure_scratch_bandwidth(directory / "no-such-directory-2311", std::uint64_t{64} << 20);
    EXPECT_EQ(none.measured_bytes, 0U);
    EXPECT_EQ(none.write_bytes_per_second, 0.0);
}

TEST(ScratchProfileTest, ASecondCallForTheSameDirectoryDoesNotMeasureAgain) {
    const std::filesystem::path directory = std::filesystem::current_path();
    const ScratchBandwidth first = scratch_bandwidth_for(directory);
    const ScratchBandwidth second = scratch_bandwidth_for(directory);
    EXPECT_GT(first.measured_bytes, 0U);
    // Bit for bit the same figures: a second measurement would not be.
    EXPECT_EQ(first.write_bytes_per_second, second.write_bytes_per_second);
    EXPECT_EQ(first.read_bytes_per_second, second.read_bytes_per_second);
    EXPECT_EQ(first.measured_bytes, second.measured_bytes);
    const HardwareProfile measured = current_hardware_profile(directory, /*measure_scratch=*/true);
    EXPECT_EQ(measured.scratch_write_bytes_per_second, first.write_bytes_per_second);
    EXPECT_EQ(measured.scratch_read_bytes_per_second, first.read_bytes_per_second);
}

TEST(ScratchProfileTest, TheReserveIsFivePercentOfTheVolumeOrEightGiB) {
    EXPECT_EQ(scratch_reserve_bytes(std::uint64_t{2} << 40), (std::uint64_t{2} << 40) / 20);
    EXPECT_EQ(scratch_reserve_bytes(std::uint64_t{100} << 30), std::uint64_t{8} << 30);
    EXPECT_EQ(scratch_reserve_bytes(std::uint64_t{160} << 30), std::uint64_t{8} << 30);
    EXPECT_EQ(scratch_reserve_bytes((std::uint64_t{160} << 30) + 20), (std::uint64_t{8} << 30) + 1);
}

}  // namespace
}  // namespace menos::plan
