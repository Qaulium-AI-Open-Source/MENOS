#include <gtest/gtest.h>

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

// Phase 21 memory accounting (docs/adr/0025). Step 1 measures and reports; it
// changes no behaviour. For every fixture of section 4:
//   * the outputs are bitwise identical to the committed baseline;
//   * the heap-tracker peak and the RSS high-water delta are printed beside the
//     estimate, as a "MEMORY" line, for section 10.1.
// The estimate >= measured assertions are added per engine in step 3.

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
              << "\n";
    RecordProperty("estimate", std::to_string(m.estimated_peak_bytes));
    RecordProperty("tracked_peak", std::to_string(m.tracked_peak_bytes));

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

TEST_P(MemoryRssTest, PeakResidentMemoryIsReported) {
    const Fixture* fixture = find_fixture(GetParam());
    ASSERT_NE(fixture, nullptr);
    if (!can_run(*fixture)) {
        GTEST_SKIP() << "no CUDA device";
    }
    const std::map<std::string, std::string> run = run_probe(fixture->name);
    ASSERT_TRUE(run.contains("rss_peak")) << "menos_memory_probe " << fixture->name << " failed";
    const std::map<std::string, std::string> base = run_probe(rss_baseline_for(*fixture).name);
    ASSERT_TRUE(base.contains("rss_peak")) << "baseline probe failed";

    if (field(run, "masked") != 0) {
        std::cout << "RSS fixture=" << fixture->name
                  << " masked=1 (the high-water mark cannot be reset here)\n";
        return;
    }
    const std::uint64_t delta = field(run, "rss_peak") - field(run, "rss_before");
    const std::uint64_t b = field(base, "rss_peak") - field(base, "rss_before");
    std::cout << "RSS fixture=" << fixture->name << " estimate=" << field(run, "estimate")
              << " rss_delta=" << delta << " b=" << b << " rss_before=" << field(run, "rss_before")
              << " hwm_reset=" << field(run, "hwm_reset") << "\n";
    RecordProperty("rss_delta", std::to_string(delta));
    RecordProperty("b", std::to_string(b));
}

INSTANTIATE_TEST_SUITE_P(Fixtures,
                         MemoryRssTest,
                         testing::ValuesIn(fixture_names(/*include_probe_only=*/true)),
                         test_name);

}  // namespace
}  // namespace menos::test::memory
