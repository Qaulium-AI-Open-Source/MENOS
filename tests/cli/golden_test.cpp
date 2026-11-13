// docs/BUILD_PLAN.md Phase 7 acceptance criterion 4: "The CLI golden-
// output tests pass for run and explain, including --statevector and
// its qubit-count refusal." Exercises menos::cli::run() -- the exact
// same command-dispatch and formatting logic the built `menos` binary
// runs -- in process, so no subprocess-spawning/platform-portability
// concerns are needed to test it (cli/cli_impl.h).

#include "cli_impl.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace menos::cli {
namespace {

// name/text cannot be usefully confused at any call site below -- name
// is always a short identifier literal, text is always a multi-line
// R"(...)" QASM2 block -- but they are the same type, so clang-tidy
// still flags the pair; NOLINT rather than a wrapper type for two
// plain strings in test-only code.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::string write_temp_qasm(const std::string& name, const std::string& text) {
    const std::string path = std::string(testing::TempDir()) + "menos_cli_golden_" + name + ".qasm";
    std::ofstream out(path);
    out << text;
    out.close();
    return path;
}

// The two wall-clock fields (execute_s, sample_s) are, by construction,
// not deterministic; blank them before comparing (matching
// tests/api/explain_test.cpp's own normalize_timings precedent).
std::string normalize_timings(const std::string& text) {
    static const std::regex kTimingPattern(R"((execute_s|sample_s|plan_s|replan_s)=[^ \n]+)");
    return std::regex_replace(text, kTimingPattern, "$1=<t>");
}

// Byte figures are deterministic on one build but not across builds: since
// Phase 21 they count the standard library's own container sizes (which
// differ between libstdc++, the MSVC STL and MSVC's checked-iterator Debug
// mode) and the LAPACK's SVD workspace (docs/adr/0025 section 3.4). They are
// pinned by each engine's estimate tests instead; here they are blanked.
std::string normalize_bytes(const std::string& text) {
    static const std::regex kBytesPattern(R"(([a-z_]*bytes)=[0-9]+)");
    return std::regex_replace(text, kBytesPattern, "$1=<b>");
}

constexpr const char* kBellQasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
)";

TEST(CliGoldenTest, ExplainProducesExactDeterministicOutput) {
    const std::string path = write_temp_qasm("explain_bell", kBellQasm);
    std::ostringstream out;
    std::ostringstream err;
    const int code = run({"explain", path, "--shots", "100", "--seed", "1"}, out, err);

    EXPECT_EQ(code, 0);
    EXPECT_EQ(err.str(), "");
    // Plan::explain() has no wall-clock content at all -- explain()'s
    // entire output is deterministic, so this is a true exact-string
    // comparison, not a normalized one.
    const std::string expected =
        "MENOS EXECUTION REPORT\n"
        "circuit    qubits=2 clbits=2 operations=4 depth=3 global_phase=0\n"
        "light_cone qubits 2->2 operations 4->4\n"
        "candidate 0strategy=direct_dense feasible=true winner=false peak_bytes=<b> "
        "work_units=24\n"
        "candidate 1strategy=direct_stabilizer feasible=true winner=true peak_bytes=<b> "
        "work_units=16\n"
        "candidate 2strategy=direct_mps feasible=true winner=false peak_bytes=<b> "
        "work_units=34\n"
        "strategy   direct_stabilizer (exact)\n"
        "region 0   qubits=[0,1] engine=stabilizer backend=cpu\n"
        "estimated  state_bytes=<b> work_units=16\n"
        "\n";
    EXPECT_EQ(normalize_bytes(out.str()), expected);
}

TEST(CliGoldenTest, ExplainWithoutFlagsUsesTheDefaultsAndStillProducesAPlan) {
    const std::string path = write_temp_qasm("explain_bell_no_flags", kBellQasm);
    std::ostringstream out;
    std::ostringstream err;
    const int code = run({"explain", path}, out, err);

    EXPECT_EQ(code, 0);
    EXPECT_EQ(err.str(), "");
    EXPECT_NE(out.str().find("MENOS EXECUTION REPORT\n"), std::string::npos);
    EXPECT_NE(out.str().find("strategy   direct_stabilizer (exact)\n"), std::string::npos);
}

TEST(CliGoldenTest, RunProducesExactOutputAfterNormalizingTimings) {
    const std::string path = write_temp_qasm("run_bell", kBellQasm);
    std::ostringstream out;
    std::ostringstream err;
    const int code = run({"run", path, "--shots", "100", "--seed", "1"}, out, err);

    EXPECT_EQ(code, 0);
    EXPECT_EQ(err.str(), "");
    const std::string expected =
        "MENOS EXECUTION REPORT\n"
        "circuit    qubits=2 clbits=2 operations=4 depth=3 global_phase=0\n"
        "light_cone qubits 2->2 operations 4->4\n"
        "candidate 0strategy=direct_dense feasible=true winner=false peak_bytes=<b> "
        "work_units=24\n"
        "candidate 1strategy=direct_stabilizer feasible=true winner=true peak_bytes=<b> "
        "work_units=16\n"
        "candidate 2strategy=direct_mps feasible=true winner=false peak_bytes=<b> "
        "work_units=34\n"
        "strategy   direct_stabilizer (exact)\n"
        "region 0   qubits=[0,1] engine=stabilizer backend=cpu\n"
        "estimated  state_bytes=<b> work_units=16\n"
        "actual     state_bytes=<b> execute_s=<t> sample_s=<t> plan_s=<t> peak_host_bytes=<b> "
        "peak_device_bytes=<b>\n"
        "00: 50\n"
        "11: 50\n";
    EXPECT_EQ(normalize_bytes(normalize_timings(out.str())), expected);
}

TEST(CliGoldenTest, RunWithStatevectorOnASmallCircuitPrintsAmplitudes) {
    constexpr const char* kOneQubitQasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
)";
    const std::string path = write_temp_qasm("run_statevector_small", kOneQubitQasm);
    std::ostringstream out;
    std::ostringstream err;
    const int code = run({"run", path, "--shots", "0", "--seed", "0", "--statevector"}, out, err);

    EXPECT_EQ(code, 0);
    EXPECT_EQ(err.str(), "");
    EXPECT_NE(out.str().find("0: 0.707106781187+0i\n"), std::string::npos) << out.str();
    EXPECT_NE(out.str().find("1: 0.707106781187+0i\n"), std::string::npos) << out.str();
}

TEST(CliGoldenTest, StatevectorAboveTheQubitLimitFailsWithTheDocumentedMessage) {
    std::string qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[11];\n";
    for (int i = 0; i < 11; ++i) {
        qasm += "h q[" + std::to_string(i) + "];\n";
    }
    const std::string path = write_temp_qasm("run_statevector_too_big", qasm);
    std::ostringstream out;
    std::ostringstream err;
    const int code = run({"run", path, "--shots", "0", "--seed", "0", "--statevector"}, out, err);

    EXPECT_EQ(code, 1);
    EXPECT_EQ(out.str(), "");
    EXPECT_NE(err.str().find("11-qubit"), std::string::npos) << err.str();
    EXPECT_NE(err.str().find("10-qubit"), std::string::npos) << err.str();
    EXPECT_NE(err.str().find("Python API"), std::string::npos) << err.str();
}

TEST(CliGoldenTest, RunWithoutShotsOrSeedFailsAsUsageError) {
    const std::string path = write_temp_qasm("run_missing_flags", kBellQasm);
    {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(run({"run", path, "--seed", "1"}, out, err), 1);
        EXPECT_NE(err.str().find("--shots"), std::string::npos) << err.str();
    }
    {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(run({"run", path, "--shots", "10"}, out, err), 1);
        EXPECT_NE(err.str().find("--seed"), std::string::npos) << err.str();
    }
}

TEST(CliGoldenTest, UnknownCommandFailsAsUsageError) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run({"frobnicate", "file.qasm"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown command"), std::string::npos) << err.str();
}

// Phase 17 (docs/adr/0021 section 4): --time-budget. With a budget, explain()
// prints machine-dependent est_seconds figures, so these tests check the
// budget's presence and the outcome rather than an exact string. The exact
// goldens above, which pass no budget, are unchanged.
TEST(CliGoldenTest, ExplainWithATimeBudgetReportsIt) {
    const std::string path = write_temp_qasm("explain_bell_budget", kBellQasm);
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run({"explain", path, "--shots", "100", "--time-budget", "3600"}, out, err), 0);
    EXPECT_EQ(err.str(), "");
    EXPECT_NE(out.str().find("time_budget"), std::string::npos) << out.str();
}

TEST(CliGoldenTest, RunWithAGenerousTimeBudgetGivesTheSameCounts) {
    const std::string path = write_temp_qasm("run_bell_budget", kBellQasm);
    std::ostringstream plain_out;
    std::ostringstream plain_err;
    ASSERT_EQ(run({"run", path, "--shots", "100", "--seed", "1"}, plain_out, plain_err), 0);
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(
        run({"run", path, "--shots", "100", "--seed", "1", "--time-budget", "3600"}, out, err), 0);
    // The counts lines follow the report and must be identical. 100 Bell shots
    // with seed 1 include "00" (the exact golden above has 50).
    const auto counts_of = [](const std::string& text) { return text.substr(text.find("\n00: ")); };
    ASSERT_NE(plain_out.str().find("\n00: "), std::string::npos) << plain_out.str();
    EXPECT_EQ(counts_of(out.str()), counts_of(plain_out.str()));
    EXPECT_NE(out.str().find("time_budget"), std::string::npos) << out.str();
}

TEST(CliGoldenTest, AnUnmeetableTimeBudgetIsRefused) {
    const std::string path = write_temp_qasm("run_bell_tiny_budget", kBellQasm);
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(
        run({"run", path, "--shots", "100", "--seed", "7", "--time-budget", "1e-12"}, out, err), 1);
    EXPECT_NE(err.str().find("time budget"), std::string::npos) << err.str();
}

TEST(CliGoldenTest, AnInvalidTimeBudgetIsAUsageErrorNamingTheFlag) {
    const std::string path = write_temp_qasm("explain_bell_bad_budget", kBellQasm);
    for (const char* value : {"abc", "5s", "0", "-1", "nan", "inf", ""}) {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(run({"explain", path, "--time-budget", value}, out, err), 1) << value;
        EXPECT_NE(err.str().find("--time-budget"), std::string::npos) << value << ": " << err.str();
    }
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run({"explain", path, "--time-budget"}, out, err), 1);
    EXPECT_NE(err.str().find("--time-budget requires an argument"), std::string::npos) << err.str();
}

// Phase 17 (docs/adr/0021 section 2, owner Q2): --backend.
TEST(CliGoldenTest, ExplainWithBackendCpuAddsOnlyTheRequestLine) {
    const std::string path = write_temp_qasm("explain_bell_backend_cpu", kBellQasm);
    std::ostringstream plain;
    std::ostringstream plain_err;
    ASSERT_EQ(run({"explain", path, "--shots", "100", "--seed", "1"}, plain, plain_err), 0);
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(run({"explain", path, "--shots", "100", "--seed", "1", "--backend", "cpu"}, out, err),
              0);
    std::string expected = plain.str();
    const std::string after = "light_cone qubits 2->2 operations 4->4\n";
    expected.insert(expected.find(after) + after.size(), "backend    cpu (requested)\n");
    EXPECT_EQ(out.str(), expected);
}

TEST(CliGoldenTest, BackendCudaIsHonouredOrRefusedNeverRunOnTheCpu) {
    const std::string path = write_temp_qasm("run_bell_backend_cuda", kBellQasm);
    std::ostringstream out;
    std::ostringstream err;
    const int code =
        run({"run", path, "--shots", "100", "--seed", "1", "--backend", "cuda"}, out, err);
    if (code == 0) {
        EXPECT_NE(out.str().find("backend    cuda (requested)"), std::string::npos) << out.str();
        EXPECT_NE(out.str().find("engine=dense backend=cuda"), std::string::npos) << out.str();
    } else {
        EXPECT_EQ(code, 1);
        EXPECT_NE(err.str().find("backend=cuda"), std::string::npos) << err.str();
    }
}

TEST(CliGoldenTest, AnUnknownBackendIsAUsageError) {
    const std::string path = write_temp_qasm("explain_bell_bad_backend", kBellQasm);
    for (const char* value : {"gpu", "CPU", ""}) {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(run({"explain", path, "--backend", value}, out, err), 1) << value;
        EXPECT_NE(err.str().find("--backend must be 'cpu' or 'cuda'"), std::string::npos)
            << err.str();
    }
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run({"explain", path, "--backend"}, out, err), 1);
    EXPECT_NE(err.str().find("--backend requires an argument"), std::string::npos) << err.str();
}

// Phase 23 (docs/adr/0027): --tier disk runs the job on the disk tier, with
// the scratch file in --scratch-dir, and the same counts as in memory; the
// report names the request, the tier's figures and the scratch directory.
constexpr const char* kRotatedBellQasm = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
rx(0.3) q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
)";

TEST(CliGoldenTest, TierDiskRunsOnTheDiskTierWithTheSameCounts) {
    // Non-Clifford, so the in-memory run is the dense engine's and draws
    // the same counts as the disk tier.
    const std::string path = write_temp_qasm("run_bell_tier_disk", kRotatedBellQasm);
    const std::string scratch = std::filesystem::current_path().string();
    std::ostringstream disk_out;
    std::ostringstream disk_err;
    ASSERT_EQ(run({"run",
                   path,
                   "--shots",
                   "100",
                   "--seed",
                   "1",
                   "--tier",
                   "disk",
                   "--scratch-dir",
                   scratch},
                  disk_out,
                  disk_err),
              0)
        << disk_err.str();
    std::ostringstream cpu_out;
    std::ostringstream cpu_err;
    ASSERT_EQ(
        run({"run", path, "--shots", "100", "--seed", "1", "--backend", "cpu"}, cpu_out, cpu_err),
        0);
    const std::string report = disk_out.str();
    EXPECT_NE(report.find("tier       disk (requested)\n"), std::string::npos) << report;
    EXPECT_NE(report.find(" tier=disk passes=1 chunk_qubits=2 disk_bytes=64\n"), std::string::npos)
        << report;
    EXPECT_NE(
        report.find("engine=dense backend=cpu tier=disk fusion=containment blocks=1 absorbed=2\n"),
        std::string::npos)
        << report;
    EXPECT_NE(report.find(" scratch_directory=\"" + scratch + "\""), std::string::npos) << report;
    EXPECT_NE(report.find(" peak_disk_bytes=64"), std::string::npos) << report;
    // The counts, after the report, are the in-memory run's.
    ASSERT_NE(cpu_out.str().find("strategy   direct_dense (exact)"), std::string::npos)
        << cpu_out.str();
    const auto counts_of = [](const std::string& text) {
        const std::size_t first = text.find("\n00: ");
        return text.substr(first == std::string::npos ? text.find("\n01: ") : first);
    };
    EXPECT_EQ(counts_of(report), counts_of(cpu_out.str()));
    // explain previews the same plan.
    std::ostringstream explain_out;
    std::ostringstream explain_err;
    ASSERT_EQ(run({"explain", path, "--tier", "disk", "--scratch-dir", scratch},
                  explain_out,
                  explain_err),
              0)
        << explain_err.str();
    EXPECT_NE(explain_out.str().find("tier       disk (requested)\n"), std::string::npos);
    EXPECT_NE(explain_out.str().find(
                  "engine=dense backend=cpu tier=disk fusion=containment blocks=1 absorbed=2\n"),
              std::string::npos);
}

TEST(CliGoldenTest, TierInMemoryAddsOnlyTheRequestLine) {
    const std::string path = write_temp_qasm("explain_bell_tier_in_memory", kBellQasm);
    std::ostringstream plain_out;
    std::ostringstream plain_err;
    ASSERT_EQ(run({"explain", path}, plain_out, plain_err), 0);
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(run({"explain", path, "--tier", "in_memory"}, out, err), 0) << err.str();
    std::string expected = plain_out.str();
    const std::string after = "light_cone qubits 2->2 operations 4->4\n";
    expected.insert(expected.find(after) + after.size(), "tier       in_memory (requested)\n");
    EXPECT_EQ(out.str(), expected);
}

TEST(CliGoldenTest, AnUnknownTierOrAMissingScratchDirIsAUsageError) {
    const std::string path = write_temp_qasm("explain_bell_bad_tier", kBellQasm);
    for (const char* value : {"ram", "Disk", ""}) {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(run({"explain", path, "--tier", value}, out, err), 1) << value;
        EXPECT_NE(err.str().find("--tier must be 'disk' or 'in_memory'"), std::string::npos)
            << err.str();
    }
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(run({"explain", path, "--scratch-dir"}, out, err), 1);
    EXPECT_NE(err.str().find("--scratch-dir requires an argument"), std::string::npos) << err.str();
}

TEST(CliGoldenTest, ParserFailurePathSurfacesThroughTheCli) {
    const std::string path =
        write_temp_qasm("run_bad_qasm", "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n");
    std::ostringstream out;
    std::ostringstream err;
    const int code = run({"explain", path}, out, err);
    EXPECT_EQ(code, 1);
    EXPECT_NE(err.str().find("include"), std::string::npos) << err.str();
}

}  // namespace
}  // namespace menos::cli
