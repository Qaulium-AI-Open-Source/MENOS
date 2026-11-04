#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "heap_tracker.h"
#include "memory_fixtures.h"

// Phase 21 memory accounting (docs/adr/0025, the gate of section 9 Q4). For
// every fixture of section 4:
//   * the outputs are bitwise identical to the committed baseline taken before
//     any Phase 21 change;
//   * the estimate is at least the heap-tracker peak, with no slack;
//   * the estimate plus B is at least the rise in peak RSS, measured in a fresh
//     process, where B is the same figure for the smallest run of the same
//     engine in the same session;
//   * the memory ledger's peak, which Result reports (section 3.2), is at
//     most the heap-tracker peak, and its device peak at most the device
//     estimate.
// Under AddressSanitizer (the dev and windows-dev presets) the sanitizer owns
// operator new and the process's memory, so both memory checks are skipped
// there, by name; the bitwise check runs everywhere. Each figure is also
// printed ("MEMORY", "RSS") for docs/adr/0025 section 10.

namespace menos::test::memory {
namespace {

#ifdef _WIN32
constexpr const char* kBaselineFile = MENOS_MEMORY_DATA_DIR "/baseline_windows.json";
#else
constexpr const char* kBaselineFile = MENOS_MEMORY_DATA_DIR "/baseline_linux.json";
#endif

const nlohmann::json& baseline() {
    static const nlohmann::json kBaseline = [] {
        std::ifstream in(kBaselineFile);
        if (!in) {
            return nlohmann::json();
        }
        return nlohmann::json::parse(in);
    }();
    return kBaseline;
}

std::vector<std::string> fixture_names(bool include_probe_only) {
    std::vector<std::string> names;
    for (const Fixture& f : fixtures()) {
        if (f.rss_baseline || (f.probe_only && !include_probe_only)) {
            continue;
        }
        names.push_back(f.name);
    }
    return names;
}

std::string test_name(const testing::TestParamInfo<std::string>& info) {
    return info.param;
}

class MemoryFixtureTest : public testing::TestWithParam<std::string> {};

TEST_P(MemoryFixtureTest, OutputsMatchTheBaselineAndMemoryIsReported) {
    const Fixture* fixture = find_fixture(GetParam());
    ASSERT_NE(fixture, nullptr);
    if (!can_run(*fixture)) {
        GTEST_SKIP() << "no CUDA device";
    }
    const Measurement m = run_fixture(*fixture);

    std::cout << "MEMORY fixture=" << fixture->name << " strategy=" << m.strategy
              << " estimate=" << m.estimated_peak_bytes
              << " device_estimate=" << m.estimated_device_bytes
              << " reported_actual=" << m.reported_actual_bytes << " tracked_peak="
              << (heap_tracker::available() ? std::to_string(m.tracked_peak_bytes)
                                            : std::string("n/a"))
              << " ledger_host=" << m.ledger_host_bytes
              << " ledger_device=" << m.ledger_device_bytes
              << " disk_estimate=" << m.estimated_disk_bytes
              << " ledger_disk=" << m.ledger_disk_bytes << "\n";
    RecordProperty("estimate", std::to_string(m.estimated_peak_bytes));
    RecordProperty("tracked_peak", std::to_string(m.tracked_peak_bytes));
    RecordProperty("ledger_host", std::to_string(m.ledger_host_bytes));

    if (heap_tracker::available()) {
        EXPECT_LE(m.tracked_peak_bytes, m.estimated_peak_bytes)
            << fixture->name << ": the estimate is below the heap the run really used";
        // The ledger counts only buffers the run really allocated, so it can
        // never exceed what the heap tracker saw.
        EXPECT_LE(m.ledger_host_bytes, m.tracked_peak_bytes)
            << fixture->name << ": the ledger reports more than the heap the run used";
    }
    EXPECT_LE(m.ledger_host_bytes, m.estimated_peak_bytes) << fixture->name;
    EXPECT_LE(m.ledger_device_bytes, m.estimated_device_bytes) << fixture->name;
    // The disk tier's file is the whole state, charged for the run (Phase 23,
    // docs/adr/0027 section 4): the ledger's disk peak is exactly the
    // estimate's file size, and 0 for every other fixture.
    EXPECT_EQ(m.ledger_disk_bytes, m.estimated_disk_bytes) << fixture->name;
    // Every fixture allocates a state, a tableau or a term list, so a zero
    // ledger would mean an engine charges nothing.
    EXPECT_GT(m.ledger_host_bytes + m.ledger_device_bytes, 0U) << fixture->name;

    const nlohmann::json& all = baseline();
    ASSERT_TRUE(all.is_object()) << "missing baseline " << kBaselineFile;
    ASSERT_TRUE(all.contains(fixture->name)) << fixture->name << " is not in " << kBaselineFile;
    const nlohmann::json& expected = all.at(fixture->name);
    EXPECT_EQ(m.strategy, expected.at("strategy").get<std::string>());
    EXPECT_EQ(m.counts_digest, expected.at("counts").get<std::string>());
    EXPECT_EQ(m.statevector_digest, expected.at("statevector").get<std::string>());
    EXPECT_EQ(m.expectation_digest, expected.at("expectation").get<std::string>());
}

INSTANTIATE_TEST_SUITE_P(Fixtures,
                         MemoryFixtureTest,
                         testing::ValuesIn(fixture_names(/*include_probe_only=*/false)),
                         test_name);

// ---- Peak RSS, one fresh process per fixture -------------------------------

std::map<std::string, std::string> run_probe(const std::string& fixture) {
    const std::string command = std::string("\"") + MENOS_MEMORY_PROBE + "\" " + fixture;
#ifdef _WIN32
    // _popen hands the whole line to cmd /c, which strips one pair of quotes.
    FILE* pipe = _popen(("\"" + command + "\"").c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    std::map<std::string, std::string> fields;
    if (pipe == nullptr) {
        return fields;
    }
    std::string output;
    std::array<char, 512> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
#ifdef _WIN32
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    if (status != 0) {
        return fields;
    }
    std::istringstream words(output);
    std::string word;
    while (words >> word) {
        const std::size_t eq = word.find('=');
        if (eq != std::string::npos) {
            fields[word.substr(0, eq)] = word.substr(eq + 1);
        }
    }
    return fields;
}

std::uint64_t field(const std::map<std::string, std::string>& fields, const std::string& key) {
    return std::stoull(fields.at(key));
}

class MemoryRssTest : public testing::TestWithParam<std::string> {};

// A fresh process's peak RSS moves between identical runs, in steps of the
// allocator's and the kernel's granularity, independently of the job
// (measured, docs/adr/0025 section 10: one or two 64 KiB steps on CPU runs,
// and on CUDA runs of 2 to 20 qubits alike). So the fixture's figure is the
// median of kRssRuns runs, and B -- the part of a run that does not depend on
// the job -- is the largest of kRssRuns baseline runs, noise included.
constexpr std::size_t kRssRuns = 5;

struct RssFigure {
    std::uint64_t median = 0;
    std::uint64_t largest = 0;
    std::map<std::string, std::string> fields;  // the last run's report
};

// The median and largest peak-RSS rise of kRssRuns fresh runs of one fixture;
// the fields are empty if any run fails.
RssFigure rss_of(const std::string& name) {
    RssFigure figure;
    std::vector<std::uint64_t> deltas;
    for (std::size_t i = 0; i < kRssRuns; ++i) {
        figure.fields = run_probe(name);
        if (!figure.fields.contains("rss_peak")) {
            figure.fields.clear();
            return figure;
        }
        deltas.push_back(field(figure.fields, "rss_peak") - field(figure.fields, "rss_before"));
    }
    std::sort(deltas.begin(), deltas.end());
    figure.median = deltas[kRssRuns / 2];
    figure.largest = deltas.back();
    return figure;
}

TEST_P(MemoryRssTest, PeakResidentMemoryIsReported) {
    const Fixture* fixture = find_fixture(GetParam());
    ASSERT_NE(fixture, nullptr);
    if (!can_run(*fixture)) {
        GTEST_SKIP() << "no CUDA device";
    }
    if (!heap_tracker::available()) {
        GTEST_SKIP() << "AddressSanitizer build: the sanitizer's shadow memory makes RSS "
                        "meaningless here";
    }
    const RssFigure run = rss_of(fixture->name);
    ASSERT_FALSE(run.fields.empty()) << "menos_memory_probe " << fixture->name << " failed";
    const RssFigure base = rss_of(rss_baseline_for(*fixture).name);
    ASSERT_FALSE(base.fields.empty()) << "baseline probe failed";

    const std::uint64_t estimate = field(run.fields, "estimate");
    std::cout << "RSS fixture=" << fixture->name << " estimate=" << estimate
              << " rss_delta=" << run.median << " b=" << base.largest
              << " method=" << run.fields.at("method") << "\n";
    RecordProperty("rss_delta", std::to_string(run.median));
    RecordProperty("b", std::to_string(base.largest));
#ifdef _WIN32
    // Measured and reported, not asserted (the owner's decision, docs/adr/0025
    // section 9, Q4 revisited): on Windows the working set also holds the heap's
    // fragmentation under the MPS engine's growing buffers and the kernel's page
    // tables for large allocations, which no estimate of MENOS's own allocations
    // can bound. The heap-tracker check, with no slack, is asserted on Windows.
    GTEST_SKIP() << fixture->name << ": reported, not asserted, on Windows (rise " << run.median
                 << ", estimate " << estimate << ", B " << base.largest << ")";
#else
    EXPECT_LE(run.median, estimate + base.largest)
        << fixture->name << ": the process grew by more than the estimate plus B";
#endif
}

INSTANTIATE_TEST_SUITE_P(Fixtures,
                         MemoryRssTest,
                         testing::ValuesIn(fixture_names(/*include_probe_only=*/true)),
                         test_name);

}  // namespace
}  // namespace menos::test::memory
