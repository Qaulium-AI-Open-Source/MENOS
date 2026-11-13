// docs/adr/0023 (Phase 19): the CLI chooses the importer from the program's
// header and binds OpenQASM 3 `input` declarations from --input NAME=VALUE.
// In process, like golden_test.cpp.

#include "cli_impl.h"

#include <gtest/gtest.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace menos::cli {
namespace {

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::string write_temp_program(const std::string& name, const std::string& text) {
    const std::string path = std::string(testing::TempDir()) + "menos_cli_qasm3_" + name + ".qasm";
    std::ofstream out(path);
    out << text;
    out.close();
    return path;
}

std::string normalize_timings(const std::string& text) {
    static const std::regex kTimingPattern(R"((execute_s|sample_s|plan_s|replan_s)=[^ \n]+)");
    return std::regex_replace(text, kTimingPattern, "$1=<t>");
}

struct Outcome {
    int code;
    std::string out;
    std::string err;
};

Outcome invoke(const std::vector<std::string>& args) {
    std::ostringstream out;
    std::ostringstream err;
    const int code = run(args, out, err);
    return {code, normalize_timings(out.str()), err.str()};
}

constexpr const char* kBellQasm2 = R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
)";

constexpr const char* kBellQasm3 = R"(// The same circuit in OpenQASM 3.
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
qubit[2] q;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
)";

constexpr const char* kRxInput = R"(OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta;
qubit q;
bit c;
rx(theta) q;
c = measure q;
)";

TEST(CliQasm3Test, AnOpenQasm3ProgramRunsLikeTheSameOpenQasm2Program) {
    const std::string two = write_temp_program("bell2", kBellQasm2);
    const std::string three = write_temp_program("bell3", kBellQasm3);
    for (const char* command : {"run", "explain"}) {
        const Outcome a = invoke({command, two, "--shots", "200", "--seed", "7"});
        const Outcome b = invoke({command, three, "--shots", "200", "--seed", "7"});
        EXPECT_EQ(a.code, 0) << a.err;
        EXPECT_EQ(b.code, 0) << b.err;
        EXPECT_EQ(a.out, b.out) << command;
    }
}

TEST(CliQasm3Test, InputBindsTheProgramsParameter) {
    const std::string path = write_temp_program("rx_input", kRxInput);
    const Outcome flipped = invoke(
        {"run", path, "--shots", "100", "--seed", "1", "--input", "theta=3.141592653589793"});
    EXPECT_EQ(flipped.code, 0) << flipped.err;
    EXPECT_NE(flipped.out.find("\n1: 100\n"), std::string::npos) << flipped.out;

    const Outcome unchanged =
        invoke({"run", path, "--shots", "100", "--seed", "1", "--input", "theta=0"});
    EXPECT_NE(unchanged.out.find("\n0: 100\n"), std::string::npos) << unchanged.out;
}

TEST(CliQasm3Test, AMissingBindingIsAnErrorFromTheImporter) {
    const std::string path = write_temp_program("rx_unbound", kRxInput);
    const Outcome outcome = invoke({"run", path, "--shots", "10", "--seed", "1"});
    EXPECT_EQ(outcome.code, 1);
    EXPECT_NE(outcome.err.find("input 'theta' has no binding"), std::string::npos) << outcome.err;
}

TEST(CliQasm3Test, MalformedInputFlagsAreUsageErrors) {
    const std::string path = write_temp_program("rx_flags", kRxInput);
    const std::vector<std::pair<std::vector<std::string>, std::string>> cases = {
        {{"--input"}, "requires an argument"},
        {{"--input", "theta"}, "NAME=VALUE"},
        {{"--input", "=1"}, "NAME=VALUE"},
        {{"--input", "theta=abc"}, "requires a number"},
        {{"--input", "theta=inf"}, "finite"},
        {{"--input", "theta=1", "--input", "theta=2"}, "more than once"},
    };
    for (const auto& [flags, message] : cases) {
        std::vector<std::string> args = {"run", path, "--shots", "1", "--seed", "1"};
        args.insert(args.end(), flags.begin(), flags.end());
        const Outcome outcome = invoke(args);
        EXPECT_EQ(outcome.code, 1);
        EXPECT_NE(outcome.err.find(message), std::string::npos) << outcome.err;
    }
}

TEST(CliQasm3Test, AProgramWithoutAHeaderIsAUsageError) {
    const std::string path =
        write_temp_program("no_header", "include \"stdgates.inc\";\nqubit q;\nx q;\n");
    const Outcome outcome = invoke({"explain", path});
    EXPECT_EQ(outcome.code, 1);
    EXPECT_NE(outcome.err.find("no 'OPENQASM 2.0;' or 'OPENQASM 3.0;' header"), std::string::npos)
        << outcome.err;
}

TEST(CliQasm3Test, InputIsRefusedForAnOpenQasm2Program) {
    const std::string path = write_temp_program("bell2_input", kBellQasm2);
    const Outcome outcome = invoke({"explain", path, "--input", "theta=1"});
    EXPECT_EQ(outcome.code, 1);
    EXPECT_NE(outcome.err.find("--input applies only to OpenQASM 3"), std::string::npos)
        << outcome.err;
}

TEST(CliQasm3Test, AnUnsupportedConstructSurfacesThroughTheCli) {
    const std::string path = write_temp_program("reset", "OPENQASM 3.0;\nqubit q;\nreset q;\n");
    const Outcome outcome = invoke({"explain", path});
    EXPECT_EQ(outcome.code, 1);
    EXPECT_NE(outcome.err.find("'reset' is not supported"), std::string::npos) << outcome.err;
    EXPECT_NE(outcome.err.find("line 3, column 1"), std::string::npos) << outcome.err;
}

}  // namespace
}  // namespace menos::cli
