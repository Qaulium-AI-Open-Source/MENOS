#pragma once

#include <cstdint>

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
    double cuda_host_to_device_bytes_per_second = 0.0;
    double cuda_device_to_host_bytes_per_second = 0.0;
    // Fixed cost per kernel launch. The term that correctly makes a small
    // circuit slower on the GPU than on the CPU.
    double cuda_kernel_launch_seconds = 0.0;
    // Marginal device throughput, launch overhead already subtracted out.
    double cuda_device_work_units_per_second = 0.0;
};

// The default memory budget is a fixed fraction of total_memory_bytes,
// used when JobSpec.memory_budget_bytes is not set. Decided with the
// owner (not named by the spec): 50%. See
// docs/adr/0003-default-memory-budget.md.
inline constexpr double kDefaultMemoryBudgetFraction = 0.5;

// Total memory from sysconf on Linux, sysctlbyname("hw.memsize") on
// macOS, GlobalMemoryStatusEx on Windows, with both calibrated figures left
// at zero. Throws InternalError if the platform call fails.
//
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

// The profile a job should be planned against. A job without a time budget
// on a CPU-only build never reads the calibrated rates: plan_job uses them
// only to turn work units into seconds (the budget check, est_seconds in a
// budgeted explain()) and to choose the dense candidate's backend when a
// CUDA device is present. For such a job this returns the detected profile
// (memory facts only, rates zero, measured once), so the process never runs
// the calibration microbenchmarks -- whose two 32 MiB buffers were the
// largest allocation of a small job, 64 MiB of peak RSS for a figure it
// never used. A budgeted job, or any job on a CUDA build, gets
// calibrated_hardware_profile() exactly as before.
[[nodiscard]] const HardwareProfile& hardware_profile_for(bool time_budgeted);

// The measurement itself, exposed so tests can check it directly rather than
// only through the cache. Runs the microbenchmarks every call. Both are
// deadline-bounded, so this returns in roughly 30 ms on any build.
[[nodiscard]] HardwareProfile calibrate_hardware();

}  // namespace menos::plan
