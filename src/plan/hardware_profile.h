#pragma once

#include <cstdint>
#include <filesystem>

namespace menos::plan {

// Total memory bytes, plus the calibrated throughput figures Phase 10's cost
// model needs to turn work units into seconds. Every field here has a
// consumer: total_memory_bytes sets the default memory budget, and the two
// calibrated figures are what `time_budget_seconds` is evaluated against
// (docs/adr/0014).
struct HardwareProfile {
    std::uint64_t total_memory_bytes;

    // Streaming memory bandwidth, bytes per second, measured by
    // calibrate_hardware() with a DRAM-sized copy.
    //
    // Consumed by the planner as a physical floor on every candidate's
    // runtime: a candidate that materializes estimated_peak_bytes must move at
    // least that many bytes across the memory bus, so it cannot finish sooner
    // than those bytes take at this rate. The planner's time estimate is the
    // larger of that floor and the work-unit term below.
    //
    // What it bounds, stated precisely because a metric's arithmetic is not
    // the metric: it bounds the time to write the peak working set ONCE. It is
    // not a model of a run's total traffic -- a dense sweep re-reads its state
    // every gate layer and moves far more than this -- and it is therefore a
    // true lower bound rather than an estimate of the real traffic.
    double memory_bandwidth_bytes_per_second = 0.0;

    // How many cost-model work units this machine retires per second,
    // measured by running a real circuit through the real dense engine and
    // dividing its own estimated work units by the elapsed wall time. This is
    // the conversion the planner uses:
    //
    //     estimated_seconds = estimated_work_units / work_units_per_second
    //
    // Calibrating the cost model's own unit, rather than inventing a constant
    // relating flops to work units, is what keeps this honest: the number is
    // whatever this machine actually achieves on the code that will run.
    //
    // Measured on a cache-resident circuit, so it is an OPTIMISTIC bound for
    // large qubit counts, which is exactly why the bandwidth floor above is
    // applied alongside it. See plan/hardware_profile.cpp for why a
    // DRAM-sized calibration circuit was rejected.
    double work_units_per_second = 0.0;

    // --- CUDA device, all zero/false when no GPU is present or the build has
    // no CUDA support (Phase 10, docs/adr/0014). Every field is measured and
    // every field is consumed by the planner's seconds formula or its device
    // budget check; none is reported for display only.
    bool cuda_present = false;
    std::uint64_t cuda_device_total_bytes = 0;
    // The device memory budget. This is the FREE figure from cudaMemGetInfo,
    // not the total and emphatically not the nameplate: the development GPU is
    // sold as 4096 MiB, reports 3770 MiB total and 3681 MiB free.
    std::uint64_t cuda_device_available_bytes = 0;
    // The part of cuda_device_available_bytes that is MENOS's own memory
    // pool, retained from earlier jobs, and whether the pool is in use
    // (Phase 22, docs/adr/0026 section 3.4). Both are reported on the Plan.
    std::uint64_t cuda_device_pool_bytes = 0;
    bool cuda_device_pooled = false;
    double cuda_host_to_device_bytes_per_second = 0.0;
    double cuda_device_to_host_bytes_per_second = 0.0;
    // Fixed cost per kernel launch. The term that correctly makes a small
    // circuit slower on the GPU than on the CPU.
    double cuda_kernel_launch_seconds = 0.0;
    // Marginal device throughput, launch overhead already subtracted out.
    double cuda_device_work_units_per_second = 0.0;

    // --- The disk tier's scratch directory (Phase 23, docs/adr/0027 section
    // 3.3), all empty/zero until current_hardware_profile() is asked about
    // one. The directory a disk-tier job would create its scratch file in;
    // its volume's capacity and the space available on it now, read at every
    // plan like the CUDA device's free memory, so the planner's reserve
    // (scratch_reserve_bytes) and the file's size are checked against the
    // figures of the moment; and its sequential write and read bandwidth,
    // measured lazily (section 9 Q3) with at most 1 GiB written, once per
    // directory per process, and only when a job could reach the disk tier.
    // Zero bandwidth means unmeasured, and the planner treats a disk-tier
    // candidate's runtime as unknown then.
    std::filesystem::path scratch_directory = {};
    std::uint64_t scratch_capacity_bytes = 0;
    std::uint64_t scratch_available_bytes = 0;
    double scratch_write_bytes_per_second = 0.0;
    double scratch_read_bytes_per_second = 0.0;
};

// The space the disk tier leaves untouched on its volume (docs/adr/0027
// section 9, Q2): 5% of the volume's capacity or 8 GiB, whichever is larger.
[[nodiscard]] std::uint64_t scratch_reserve_bytes(std::uint64_t capacity_bytes);

// The sequential bandwidth of a scratch directory, as the disk tier moves
// data: whole chunks of 256 MiB through FileBackingStore (unbuffered where
// the file system allows), written once and read back once, on a file of
// `bytes` (rounded down to whole chunks, at most 1 GiB, at least one chunk).
// The file is gone when this returns. Zero figures when the directory cannot
// take the file (too little space beside the reserve, or not writable).
struct ScratchBandwidth {
    double write_bytes_per_second = 0.0;
    double read_bytes_per_second = 0.0;
    std::uint64_t measured_bytes = 0;
};
inline constexpr std::uint64_t kScratchMeasurementBytes = std::uint64_t{1} << 30;
[[nodiscard]] ScratchBandwidth measure_scratch_bandwidth(
    const std::filesystem::path& directory, std::uint64_t bytes = kScratchMeasurementBytes);

// measure_scratch_bandwidth(directory), measured at most once per directory
// per process and cached: the second call for the same directory returns the
// first call's figures without touching the disk.
[[nodiscard]] ScratchBandwidth scratch_bandwidth_for(const std::filesystem::path& directory);

// The default memory budget is a fixed fraction of total_memory_bytes,
// used when JobSpec.memory_budget_bytes is not set. Decided with the
// owner (not named by the spec): 50%. See
// docs/adr/0003-default-memory-budget.md.
inline constexpr double kDefaultMemoryBudgetFraction = 0.5;

// The host memory budget of a job that sets none: kDefaultMemoryBudgetFraction
// of the profile's total memory. Shared by the planner and readout
// mitigation, which refuses its own dense vectors against the same budget.
[[nodiscard]] std::uint64_t default_memory_budget(const HardwareProfile& profile);

// Total memory from sysconf on Linux, sysctlbyname("hw.memsize") on
// macOS, GlobalMemoryStatusEx on Windows, with both calibrated figures left
// at zero. Throws InternalError if the platform call fails.

// Zero means "not calibrated", and the planner treats it that way: a job with
// a time budget against an uncalibrated profile is refused rather than
// silently passed, since dividing by zero would otherwise make every
// candidate look instantaneous.
[[nodiscard]] HardwareProfile detect_hardware_profile();

// detect_hardware_profile() plus the microbenchmarks, computed once per
// process and cached. Repeated calls return the cached value without
// re-measuring, so the cost is paid at most once regardless of how many jobs
// are planned (docs/BUILD_PLAN.md Phase 10: "calibration by microbenchmark at
// initialization, with the results cached").
//
// The cache is process-lifetime and has no invalidation: a machine's
// throughput does not change while the process runs, and a persisted cache
// would need staleness rules this phase has no consumer for. Stated so the
// absence is a decision rather than an oversight.
[[nodiscard]] const HardwareProfile& calibrated_hardware_profile();

// The calibrated profile, with the CUDA device's free memory read again now
// (Phase 21, docs/adr/0025 section 9, Q3): the cached figure was taken once
// per process and could promise memory that other processes, or earlier
// jobs, have since taken. Everything else is the cached calibration. The api
// layer calls this once per job, before planning it; a replan within the job
// reuses the job's figure, so all of a job's plans are compared against the
// same device memory. Identical to calibrated_hardware_profile() when there is
// no CUDA device.
[[nodiscard]] HardwareProfile current_hardware_profile();

// current_hardware_profile() with the scratch directory's figures (Phase 23,
// docs/adr/0027 section 3.3): its capacity and available space read now, and
// its bandwidth from scratch_bandwidth_for() when measure_scratch is true
// (the api layer asks for it only when a job could reach the disk tier, so
// the one-off measurement is never paid by a job that cannot). A directory
// that cannot be read leaves every scratch figure zero, which the planner
// reports as the disk tier being out of reach.
[[nodiscard]] HardwareProfile current_hardware_profile(
    const std::filesystem::path& scratch_directory, bool measure_scratch);

// The measurement itself, exposed so tests can check it directly rather than
// only through the cache. Runs the microbenchmarks every call. Both are
// deadline-bounded, so this returns in roughly 30 ms on any build.
[[nodiscard]] HardwareProfile calibrate_hardware();

}  // namespace menos::plan