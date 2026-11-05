// menos_memory_probe: runs one memory fixture in a fresh process and reports the
// process's resident memory before the run and its highest point during it
// (docs/adr/0025 section 3.1). On Linux the kernel's high-water mark is reset
// after the one-time preparation and read after the run. Windows cannot reset
// PeakWorkingSetSize, so there a thread samples the working set continuously
// for the length of the run instead; a peak shorter than one sample can be
// missed. Built without the heap tracker, so the figures are the real
// allocator's.
//
//   menos_memory_probe <fixture>            one line of key=value pairs
//   menos_memory_probe --list               every fixture this build can run
//   menos_memory_probe --write-baseline F   writes the bitwise baseline JSON

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include <nlohmann/json.hpp>

#include "memory_fixtures.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // windows.h's min/max macros would break std::max
#endif
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <malloc.h>

#include <sstream>
#endif

namespace {

using menos::test::memory::Fixture;
using menos::test::memory::Measurement;

struct ProcessMemory {
    std::uint64_t resident_bytes = 0;
    std::uint64_t peak_resident_bytes = 0;
};

ProcessMemory process_memory() {
    ProcessMemory memory;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) != 0) {
        memory.resident_bytes = counters.WorkingSetSize;
        memory.peak_resident_bytes = counters.PeakWorkingSetSize;
    }
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t kib = 0;
        fields >> key >> kib;
        if (key == "VmRSS:") {
            memory.resident_bytes = kib * 1024;
        } else if (key == "VmHWM:") {
            memory.peak_resident_bytes = kib * 1024;
        }
    }
#endif
    return memory;
}

#ifndef _WIN32
// Linux resets VmHWM to the current RSS when "5" is written to
// /proc/self/clear_refs. Windows has no equivalent for PeakWorkingSetSize.
bool reset_high_water_mark() {
    std::ofstream clear_refs("/proc/self/clear_refs");
    clear_refs << "5";
    clear_refs.flush();
    return static_cast<bool>(clear_refs);
}
#endif

#ifdef _WIN32
// Samples the working set from its own thread, as fast as the scheduler lets
// it, until stopped, and keeps the largest value seen.
class WorkingSetSampler {
public:
    WorkingSetSampler()
        : thread_([this] {
              while (!stop_.load(std::memory_order_relaxed)) {
                  sample();
                  std::this_thread::yield();
              }
              sample();
          }) {}
    WorkingSetSampler(const WorkingSetSampler&) = delete;
    WorkingSetSampler& operator=(const WorkingSetSampler&) = delete;
    WorkingSetSampler(WorkingSetSampler&&) = delete;
    WorkingSetSampler& operator=(WorkingSetSampler&&) = delete;
    ~WorkingSetSampler() {
        stop();
    }

    std::uint64_t stop() {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) {
            thread_.join();
        }
        return peak_.load(std::memory_order_relaxed);
    }

private:
    void sample() {
        const std::uint64_t now = process_memory().resident_bytes;
        std::uint64_t seen = peak_.load(std::memory_order_relaxed);
        while (now > seen && !peak_.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
        }
    }

    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> peak_{0};
    std::thread thread_;
};
#endif

int run_one(std::string_view name) {
    const Fixture* fixture = menos::test::memory::find_fixture(name);
    if (fixture == nullptr) {
        std::cerr << "unknown fixture: " << name << "\n";
        return 2;
    }
    if (!menos::test::memory::can_run(*fixture)) {
        std::cout << "fixture=" << fixture->name << " skipped=no-device\n";
        return 0;
    }
    menos::test::memory::prepare(*fixture);
    ProcessMemory before;
#ifdef _WIN32
    std::optional<WorkingSetSampler> sampler;
    const Measurement m = menos::test::memory::run_fixture(*fixture, /*draw_samples=*/false, [&] {
        before = process_memory();
        sampler.emplace();
    });
    const std::uint64_t peak =
        std::max(sampler.has_value() ? sampler->stop() : 0, process_memory().resident_bytes);
    const char* method = "sampled";
#else
    bool reset = false;
    const Measurement m = menos::test::memory::run_fixture(*fixture, /*draw_samples=*/false, [&] {
        reset = reset_high_water_mark();
        before = process_memory();
    });
    if (!reset) {
        std::cerr << "menos_memory_probe: cannot reset the high-water mark\n";
        return 1;
    }
    const std::uint64_t peak = process_memory().peak_resident_bytes;
    const char* method = "high_water_mark";
#endif
    std::cout << "fixture=" << fixture->name << " method=" << method
              << " rss_before=" << before.resident_bytes << " rss_peak=" << peak
              << " estimate=" << m.estimated_peak_bytes
              << " device_estimate=" << m.estimated_device_bytes
              << " reported_actual=" << m.reported_actual_bytes << " strategy=" << m.strategy
              << "\n";
    return 0;
}

int write_baseline(const std::string& path) {
    nlohmann::ordered_json json = nlohmann::ordered_json::object();
    for (const Fixture& fixture : menos::test::memory::fixtures()) {
        if (fixture.probe_only || !menos::test::memory::can_run(fixture)) {
            continue;
        }
        const Measurement m = menos::test::memory::run_fixture(fixture);
        json[fixture.name] = {{"strategy", m.strategy},
                              {"counts", m.counts_digest},
                              {"statevector", m.statevector_digest},
                              {"expectation", m.expectation_digest}};
    }
    std::ofstream out(path);
    out << json.dump(2) << "\n";
    return out ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
    // glibc raises its mmap threshold each time a large block is freed, up to
    // 32 MiB, so after a warm-up frees a big buffer, the run's own large
    // buffers would come from the fragmenting brk heap instead and its RSS
    // would depend on what ran before it. Pinning the threshold at glibc's own
    // default, 128 KiB, keeps every run's large buffers mmapped, each released
    // whole when freed (docs/adr/0025 section 10).
    mallopt(M_MMAP_THRESHOLD, 128 * 1024);
#endif
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--list") {
            for (const Fixture& fixture : menos::test::memory::fixtures()) {
                std::cout << fixture.name << "\n";
            }
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--write-baseline") {
            return write_baseline(argv[2]);
        }
        if (argc == 2) {
            return run_one(argv[1]);
        }
        std::cerr << "usage: menos_memory_probe <fixture> | --list | --write-baseline FILE\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "menos_memory_probe: " << e.what() << "\n";
        return 1;
    }
}
