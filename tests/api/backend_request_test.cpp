// Phase 17 (docs/adr/0021 section 2): RunOptions::backend, in every build.
//
// The CUDA-only behaviour (a forced backend that actually moves a region onto
// the device, and the refusals that depend on a calibrated device) is in
// tests/plan/cuda_backend_selection_test.cpp and cuda_backend_request_test.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <regex>
#include <string>
#include <vector>

#include "common/error.h"
#include "menos/menos.h"

namespace menos {
namespace {

Circuit bell() {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    return circuit;
}

// The Bell report as it was before Phase 17 (the same text as the CLI golden in
// tests/cli/golden_test.cpp). A job that sets no backend must still produce
// exactly this.
// Byte figures are blanked: since Phase 21 they count the standard library's
// container sizes and the LAPACK's SVD workspace, which differ between builds
// (docs/adr/0025 section 3.4). The figures themselves are compared exactly
// below, between the forced and unforced plans of the same build.
constexpr const char* kBellReport =
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
    "estimated  state_bytes=<b> work_units=16\n";

std::string blank_bytes(const std::string& text) {
    static const std::regex kBytesPattern(R"(([a-z_]*bytes)=[0-9]+)");
    return std::regex_replace(text, kBytesPattern, "$1=<b>");
}

TEST(BackendRequestTest, AnUnsetBackendLeavesTheReportByteIdentical) {
    const RunOptions options{.shots = 100, .seed = 1};
    ASSERT_FALSE(options.backend.has_value());
    EXPECT_EQ(blank_bytes(preview(bell(), options).explain()), kBellReport);
}

TEST(BackendRequestTest, ForcedCpuAddsOnlyTheRequestLineAndChangesNothingElse) {
    RunOptions options{.shots = 100, .seed = 1};
    options.backend = Backend::Cpu;
    // Exact, figures included: the unforced report of this same build, plus
    // the request line.
    std::string expected = preview(bell(), RunOptions{.shots = 100, .seed = 1}).explain();
    const std::string after = "light_cone qubits 2->2 operations 4->4\n";
    expected.insert(expected.find(after) + after.size(), "backend    cpu (requested)\n");
    EXPECT_EQ(preview(bell(), options).explain(), expected);

    const Result forced = run(bell(), options);
    const Result unforced = run(bell(), RunOptions{.shots = 100, .seed = 1});
    EXPECT_EQ(forced.counts(), unforced.counts());
    ASSERT_EQ(forced.plan().regions.size(), 1U);
    EXPECT_EQ(forced.plan().regions[0].backend, Backend::Cpu);
}

TEST(BackendRequestTest, ForcedCudaOnANoisyJobIsUnsupported) {
    // In a CPU build this is refused because there is no CUDA support at all;
    // in a CUDA build, because the CUDA backend is noiseless. Either way it is
    // UnsupportedFeature and never a run on the CPU.
    Circuit circuit(1, 1);
    circuit.append(Opcode::H, {0});
    const double keep = std::sqrt(0.9);
    const double flip = std::sqrt(0.1);
    circuit.append_channel({0},
                           {{{keep, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {keep, 0.0}},
                            {{0.0, 0.0}, {flip, 0.0}, {flip, 0.0}, {0.0, 0.0}}});
    circuit.append_measure(0, 0);
    RunOptions options{.shots = 10, .seed = 1};
    options.backend = Backend::Cuda;
    EXPECT_THROW(static_cast<void>(run(circuit, options)), common::UnsupportedFeature);
}

#ifndef MENOS_ENABLE_CUDA
TEST(BackendRequestTest, ForcedCudaInABuildWithoutCudaIsUnsupported) {
    RunOptions options{.shots = 10, .seed = 1};
    options.backend = Backend::Cuda;
    try {
        static_cast<void>(run(bell(), options));
        FAIL() << "expected UnsupportedFeature";
    } catch (const common::UnsupportedFeature& error) {
        EXPECT_NE(std::string(error.what()).find("no CUDA support"), std::string::npos)
            << error.what();
    }
}
#endif

}  // namespace
}  // namespace menos
