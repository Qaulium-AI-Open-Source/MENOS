#include "plan/hardware_profile.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#error "menos supports only Linux, macOS and Windows (docs/adr/0022)"
#endif

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstring>
#include <map>
#include <mutex>
#include <span>
#include <system_error>
#include <vector>

#include "common/error.h"
#include "common/thread_count.h"
#include "cost/cost_model.h"
#include "engines/dense/dense_engine.h"
#include "engines/dense_tiered/chunked_executor.h"
#include "engines/dense_tiered/file_backing_store.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"
#include "plan/time_model.h"

#ifdef MENOS_ENABLE_CUDA
#include "kernels/cuda/device_info.h"
#include "kernels/cuda/device_pool.h"
#include "plan/cuda_calibration.h"
#endif

namespace menos::plan {
namespace {

using Clock = std::chrono::steady_clock;

// Methodology, recorded here because a measurement whose method is undocumented
// is not reproducible (docs/adr/0014).
//
// Both microbenchmarks are deadline-bounded rather than fixed-iteration: each
// runs a minimum of one repetition, then repeats only while it is still under
// its deadline. That keeps the cost of calibration bounded by construction on
// every build configuration -- a Debug+sanitizer build is roughly an order of
// magnitude slower than Release, and a fixed iteration count tuned for one is
// unusable in the other. What the sanitizer build then measures is genuinely
// its own (slower) throughput, which is the correct answer for that build.
constexpr double kBandwidthDeadlineSeconds = 0.015;
constexpr std::uint64_t kBandwidthMinRepetitions = 2;
constexpr double kThroughputDeadlineSeconds = 0.015;
constexpr std::uint64_t kThroughputMinRepetitions = 2;

// Bandwidth: memcpy between two buffers whose combined footprint (64 MiB)
// exceeds any current laptop or desktop last-level cache, so the traffic
// genuinely reaches DRAM instead of being served from cache. memcpy rather
// than a hand-written element loop for two reasons: it is the standard
// streaming-bandwidth measurement, and a sanitizer instruments it as a single
// interceptor call rather than per element, so the figure is not dominated by
// instrumentation overhead that has nothing to do with the memory system.
constexpr std::size_t kBandwidthBufferBytes = 32UL * 1024 * 1024;

// Throughput: a real circuit through the real dense engine.
//
// 13 qubits is a 128 KiB statevector: above per-core L1 on every supported
// machine, so the figure is not a pure L1 artefact, and small enough that the
// whole of calibrate_hardware() costs well under a second even on a
// Debug+sanitizer build.
//
// LIMITATION, stated rather than buried: 128 KiB is cache-resident. The figure
// this yields is therefore an OPTIMISTIC upper bound on throughput -- a
// 28-qubit run streams 4 GiB per gate layer and cannot reach it. That is
// deliberate, and is why the planner also applies a memory-traffic floor
// derived from memory_bandwidth_bytes_per_second (see hardware_profile.h and
// docs/adr/0014): the cache-resident figure bounds the work, the bandwidth
// figure bounds the traffic, and the estimate is the larger of the two.
//
// A DRAM-sized calibration circuit was rejected on measurement, not by
// assumption: the rate varies by under 1% across 12, 13, 14 and 16 qubits, so
// a larger one buys no accuracy while costing seconds per process under a
// sanitizer.
constexpr std::uint32_t kThroughputQubits = 13;
constexpr std::uint32_t kThroughputGateLayers = 2;

double measure_memory_bandwidth() {
    const std::size_t count = kBandwidthBufferBytes;
    std::vector<char> source(count, 1);
    std::vector<char> destination(count, 0);

    // One copy is measured at a time, so the loop's quantum is a whole 32 MiB
    // copy (a few milliseconds). Two consequences are handled explicitly,
    // because getting either wrong was observed to move the result by 6x:
    //
    //   - The first timed copy can be far slower than steady state (page
    //     faults, cold TLB, frequency ramp). A single warm-up copy was NOT
    //     enough: in a Release build the first timed copy alone exceeded the
    //     deadline, the loop stopped at one repetition, and the reported
    //     figure came out ~6x below the same machine's steady-state rate. So
    //     the entire deadline loop runs once and its result is discarded.
    //
    //   - A deadline alone quantizes badly when a single repetition already
    //     exceeds it, which is the normal case on a sanitizer build.
    //     kBandwidthMinRepetitions then guarantees at least two, so one
    //     outlier cannot be the whole measurement.
    //
    // The volatile read inside the loop is load-bearing, not decoration: each
    // memcpy writes a buffer nothing else observes, so at -O3 GCC is entitled
    // to delete every one of them as a dead store. It does. Measured with the
    // sink outside the loop, the warm pass reported 558691 copies in 15 ms --
    // 2.5 PB/s, i.e. an empty loop -- which silently defeated the warm-up and
    // left the measured pass running cold at half the machine's real rate.
    // Reading the destination through a volatile pointer after each copy makes
    // that copy observable, so none can be elided.
    const volatile char* const observer = destination.data();
    char sink = 0;

    const auto timed_pass = [&]() {
        std::uint64_t repetitions = 0;
        const Clock::time_point start = Clock::now();
        double seconds = 0.0;
        do {
            std::memcpy(destination.data(), source.data(), count);
            sink = static_cast<char>(sink ^ observer[count / 2]);
            ++repetitions;
            seconds = std::chrono::duration<double>(Clock::now() - start).count();
        } while (seconds < kBandwidthDeadlineSeconds || repetitions < kBandwidthMinRepetitions);
        // Each repetition reads count bytes and writes count bytes.
        const double bytes = static_cast<double>(count) * 2.0 * static_cast<double>(repetitions);
        return seconds > 0.0 ? bytes / seconds : 0.0;
    };

    static_cast<void>(timed_pass());  // discarded: warms pages, TLB and clocks
    const double result = timed_pass();
    static_cast<void>(sink);
    return result;
}

Circuit make_calibration_circuit(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, 0);
    for (std::uint32_t layer = 0; layer < kThroughputGateLayers; ++layer) {
        for (std::uint32_t q = 0; q < num_qubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

// The dense rate on `num_qubits` qubits with up to `max_threads` threads.
// At kThroughputQubits the kernels run on one thread whatever max_threads is
// (2^12 groups, below kParallelMinGroups); at kThreadedThroughputQubits they
// use every thread given (docs/adr/0028 section 16.10.3).
// A qubit count and a thread cap: distinct roles, named at both call sites.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double measure_work_units_per_second(std::uint32_t num_qubits, std::uint32_t max_threads) {
    const Circuit circuit = make_calibration_circuit(num_qubits);
    const engines::dense::DenseEngine engine;

    // The work units the cost model itself assigns to this circuit. Dividing
    // by measured time calibrates the cost model's own unit against this
    // machine, which is the whole point: no constant relating work units to
    // seconds is invented anywhere.
    const EngineEstimateRequest estimate_request{
        circuit.num_qubits(),
        &circuit.operations(),
        /*want_statevector=*/true,
        /*want_counts=*/false,
        /*want_expectation=*/false,
        /*observable=*/nullptr,
    };
    const CostEstimate estimate = engine.estimate(estimate_request);
    if (!estimate.feasible || estimate.estimated_work_units == 0) {
        return 0.0;
    }

    const EngineExecutionRequest request{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = true,
        .want_counts = false,
        .want_expectation = false,
        .shots = 0,
        .measure_targets = {},
        .bitstring_width = 0,
        .seed = 0,
        .global_phase = 0.0,
        .budget_bytes = estimate.estimated_peak_bytes,
        .is_sole_region = true,
        .observable = nullptr,
        .max_threads = max_threads,
    };

    // Same discarded-pass and minimum-repetition policy as the bandwidth
    // measurement above, for the same reason.
    const auto timed_pass = [&]() {
        std::uint64_t repetitions = 0;
        const Clock::time_point start = Clock::now();
        double seconds = 0.0;
        do {
            static_cast<void>(engine.execute(request));
            ++repetitions;
            seconds = std::chrono::duration<double>(Clock::now() - start).count();
        } while (seconds < kThroughputDeadlineSeconds || repetitions < kThroughputMinRepetitions);
        return seconds > 0.0 ? static_cast<double>(estimate.estimated_work_units) *
                                   static_cast<double>(repetitions) / seconds
                             : 0.0;
    };

    static_cast<void>(timed_pass());  // discarded
    return timed_pass();
}

}  // namespace

HardwareProfile detect_hardware_profile() {
#if defined(__APPLE__)
    std::uint64_t mem = 0;
    std::size_t len = sizeof(mem);
    const int rc = sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
    common::ensure(rc == 0, "detect_hardware_profile: sysctlbyname(\"hw.memsize\") failed");
    return HardwareProfile{mem, 0.0, 0.0};
#elif defined(__linux__)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    common::ensure(pages > 0 && page_size > 0, "detect_hardware_profile: sysconf failed");
    return HardwareProfile{
        static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size), 0.0, 0.0};
#elif defined(_WIN32)
    // ullTotalPhys is the physical memory the OS can use: the same quantity
    // as Linux's _SC_PHYS_PAGES * _SC_PAGE_SIZE (docs/adr/0022 section 2).
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    common::ensure(GlobalMemoryStatusEx(&status) != 0,
                   "detect_hardware_profile: GlobalMemoryStatusEx failed");
    return HardwareProfile{static_cast<std::uint64_t>(status.ullTotalPhys), 0.0, 0.0};
#endif
}

HardwareProfile calibrate_hardware() {
    HardwareProfile profile = detect_hardware_profile();
    profile.memory_bandwidth_bytes_per_second = measure_memory_bandwidth();
    profile.work_units_per_second =
        measure_work_units_per_second(kThroughputQubits, common::default_thread_count());
    // Every CPU engine timed in its own unit (Phase 24, docs/adr/0028
    // section 20).
    profile.seconds_per_time_count = calibrate_time_model();
#ifdef MENOS_ENABLE_CUDA
    // Only measured when the build has CUDA support AND a device answers. A
    // CUDA-enabled binary on a machine without a GPU leaves every field zero
    // and therefore behaves exactly like a CUDA-less one.
    const CudaCalibration cuda = measure_cuda_calibration();
    profile.cuda_present = cuda.present;
    profile.cuda_device_total_bytes = cuda.device_total_bytes;
    profile.cuda_device_available_bytes = cuda.device_available_bytes;
    profile.cuda_host_to_device_bytes_per_second = cuda.host_to_device_bytes_per_second;
    profile.cuda_device_to_host_bytes_per_second = cuda.device_to_host_bytes_per_second;
    profile.cuda_kernel_launch_seconds = cuda.kernel_launch_seconds;
    profile.cuda_execution_seconds = cuda.execution_seconds;
    profile.cuda_device_work_units_per_second = cuda.device_work_units_per_second;
#endif
    return profile;
}

const HardwareProfile& hardware_profile_for(bool time_budgeted) {
#ifdef MENOS_ENABLE_CUDA
    static_cast<void>(time_budgeted);
    return calibrated_hardware_profile();
#else
    if (time_budgeted) {
        return calibrated_hardware_profile();
    }
    static const HardwareProfile kDetected = detect_hardware_profile();
    return kDetected;
#endif
}

std::uint64_t default_memory_budget(const HardwareProfile& profile) {
    return static_cast<std::uint64_t>(static_cast<double>(profile.total_memory_bytes) *
                                      kDefaultMemoryBudgetFraction);
}

namespace {

HardwareProfile with_current_device_figures(HardwareProfile profile) {
#ifdef MENOS_ENABLE_CUDA
    if (profile.cuda_present) {
        // Free memory plus what MENOS's own pool already holds for reuse
        // (Phase 22, docs/adr/0026 section 3.4).
        profile.cuda_device_available_bytes = kernels::cuda::device_available_bytes();
        profile.cuda_device_pool_bytes = kernels::cuda::device_pool_bytes();
        profile.cuda_device_pooled = kernels::cuda::device_pool().pooled();
    }
#endif
    return profile;
}

}  // namespace

HardwareProfile current_hardware_profile() {
    return with_current_device_figures(calibrated_hardware_profile());
}

std::uint64_t scratch_reserve_bytes(std::uint64_t capacity_bytes) {
    constexpr std::uint64_t kEightGiB = std::uint64_t{8} << 30;
    return std::max(capacity_bytes / 20, kEightGiB);
}

ScratchBandwidth measure_scratch_bandwidth(const std::filesystem::path& directory,
                                           std::uint64_t bytes) {
    using engines::dense_tiered::FileBackingStore;
    using engines::dense_tiered::Window;
    // Chunks of 2^24 amplitudes, 256 MiB: the smallest size the disk tier
    // prefers and the smallest step 1 measured (docs/adr/0027 section 10.1).
    constexpr std::uint32_t kChunkQubits = 24;
    constexpr std::uint64_t kChunkBytes = (std::uint64_t{1} << kChunkQubits) * 16;
    ScratchBandwidth result;
    const std::uint64_t chunks =
        std::clamp<std::uint64_t>(std::min(bytes, kScratchMeasurementBytes) / kChunkBytes, 1, 4);
    const std::uint64_t file_bytes = chunks * kChunkBytes;
    std::error_code error;
    const std::filesystem::space_info info = std::filesystem::space(directory, error);
    if (error || info.available < file_bytes + scratch_reserve_bytes(info.capacity)) {
        return result;
    }
    try {
        const FileBackingStore store(
            directory, chunks << kChunkQubits, std::uint64_t{1} << kChunkQubits);
        Window window(kChunkQubits);
        const std::span<std::complex<double>> chunk = window.slot(0, store.chunk_amplitudes());
        // Resident and non-zero, so a write moves real bytes.
        for (std::size_t i = 0; i < chunk.size(); i += 256) {
            chunk[i] = std::complex<double>{static_cast<double>(i), 1.0};
        }
        const auto start_write = std::chrono::steady_clock::now();
        for (std::uint64_t c = 0; c < chunks; ++c) {
            store.write(c, chunk);
        }
        const double write_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_write).count();
        const auto start_read = std::chrono::steady_clock::now();
        for (std::uint64_t c = 0; c < chunks; ++c) {
            store.read(c, chunk);
        }
        const double read_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_read).count();
        if (write_seconds > 0.0 && read_seconds > 0.0) {
            result.write_bytes_per_second = static_cast<double>(file_bytes) / write_seconds;
            result.read_bytes_per_second = static_cast<double>(file_bytes) / read_seconds;
            result.measured_bytes = file_bytes;
        }
    } catch (const common::Error&) {
        // Not writable, or no longer roomy: unmeasured, and the planner says
        // so rather than guessing.
        result = ScratchBandwidth{};
    }
    return result;
}

ScratchBandwidth scratch_bandwidth_for(const std::filesystem::path& directory) {
    // One measurement per directory per process (docs/adr/0027 section 9,
    // Q3): the cache is written under the mutex and never invalidated, like
    // the calibration cache below.
    static std::mutex mutex;
    static std::map<std::filesystem::path, ScratchBandwidth> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(directory);
    if (found != cache.end()) {
        return found->second;
    }
    const ScratchBandwidth measured = measure_scratch_bandwidth(directory);
    cache.emplace(directory, measured);
    return measured;
}

HardwareProfile current_hardware_profile(const std::filesystem::path& scratch_directory,
                                         bool measure_scratch) {
    return current_hardware_profile(
        calibrated_hardware_profile(), scratch_directory, measure_scratch);
}

HardwareProfile current_hardware_profile(const HardwareProfile& base,
                                         const std::filesystem::path& scratch_directory,
                                         bool measure_scratch) {
    HardwareProfile profile = with_current_device_figures(base);
    std::error_code error;
    const std::filesystem::space_info info = std::filesystem::space(scratch_directory, error);
    if (error) {
        return profile;
    }
    profile.scratch_capacity_bytes = info.capacity;
    profile.scratch_available_bytes = info.available;
    if (measure_scratch) {
        const ScratchBandwidth bandwidth = scratch_bandwidth_for(scratch_directory);
        profile.scratch_write_bytes_per_second = bandwidth.write_bytes_per_second;
        profile.scratch_read_bytes_per_second = bandwidth.read_bytes_per_second;
        profile.dense_threaded_work_units_per_second = dense_threaded_work_units_per_second();
    }
    return profile;
}

double dense_threaded_work_units_per_second() {
    // Function-local static, like the calibration below: measured once, on
    // first use, then reused for the process lifetime.
    static const double kRate =
        measure_work_units_per_second(kThreadedThroughputQubits, common::default_thread_count());
    return kRate;
}

const HardwareProfile& calibrated_hardware_profile() {
    // Function-local static: measured once, on first use, then reused for the
    // process lifetime. Not the "global mutable state" CLAUDE.md forbids --
    // it is written once during initialization and read-only thereafter.
    static const HardwareProfile kProfile = calibrate_hardware();
    return kProfile;
}

}  // namespace menos::plan
