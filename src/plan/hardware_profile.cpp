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

#include <chrono>
#include <cstring>
#include <vector>

#include "common/error.h"
#include "cost/cost_model.h"
#include "engines/dense/dense_engine.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"

#ifdef MENOS_ENABLE_CUDA
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

Circuit make_calibration_circuit() {
    Circuit circuit(kThroughputQubits, 0);
    for (std::uint32_t layer = 0; layer < kThroughputGateLayers; ++layer) {
        for (std::uint32_t q = 0; q < kThroughputQubits; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(q)});
        }
        for (std::uint32_t q = layer % 2; q + 1 < kThroughputQubits; q += 2) {
            circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
        }
    }
    return circuit;
}

double measure_work_units_per_second() {
    const Circuit circuit = make_calibration_circuit();
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
    profile.work_units_per_second = measure_work_units_per_second();
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

const HardwareProfile& calibrated_hardware_profile() {
    // Function-local static: measured once, on first use, then reused for the
    // process lifetime. Not the "global mutable state" CLAUDE.md forbids --
    // it is written once during initialization and read-only thereafter.
    static const HardwareProfile kProfile = calibrate_hardware();
    return kProfile;
}

}  // namespace menos::plan
