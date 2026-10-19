// menos_memory_probe: runs one memory fixture in a fresh process and reports the
// process's resident memory before the run and its high-water mark after it
// (docs/adr/0025 section 3.1). A fresh process needs no reset of the high-water
// mark, which Windows cannot reset in-process. Built without the heap tracker,
// so the figures are the real allocator's.
//
//   menos_memory_probe <fixture>            one line of key=value pairs
//   menos_memory_probe --list               every fixture this build can run
//   menos_memory_probe --write-baseline F   writes the bitwise baseline JSON

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "memory_fixtures.h"

#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
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

// Linux resets VmHWM to the current RSS when "5" is written to
// /proc/self/clear_refs. Windows has no equivalent for PeakWorkingSetSize.
bool reset_high_water_mark() {
#ifdef _WIN32
    return false;
#else
    std::ofstream clear_refs("/proc/self/clear_refs");
    clear_refs << "5";
    clear_refs.flush();
    return static_cast<bool>(clear_refs);
#endif
}

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
    const bool reset = reset_high_water_mark();
    const ProcessMemory before = process_memory();
    const Measurement m = menos::test::memory::run_fixture(*fixture);
    const ProcessMemory after = process_memory();
    // Without a reset, a peak no higher than the one before the run belongs to
    // the preparation (or startup), and the run's own peak is not visible.
    const bool masked = !reset && after.peak_resident_bytes <= before.peak_resident_bytes;
    std::cout << "fixture=" << fixture->name << " hwm_reset=" << (reset ? 1 : 0)
              << " masked=" << (masked ? 1 : 0) << " rss_before=" << before.resident_bytes
              << " rss_peak=" << after.peak_resident_bytes << " estimate=" << m.estimated_peak_bytes
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
