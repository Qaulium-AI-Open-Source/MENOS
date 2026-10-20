// Phase 21 (docs/adr/0025 section 3.2): Result::peak_host_bytes() and
// peak_device_bytes(), the memory ledger's peaks for the successful attempt,
// and their place on explain()'s "actual" line. The ledger counts the bytes
// each buffer asked for, so on these hand-countable jobs the figure is exact.
// tests/memory checks it against the heap tracker and the estimate on every
// engine; tests/api/cuda_peak_memory_test.cpp covers the device figure.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <format>
#include <string>
#include <utility>

#include "menos/bitstring.h"
#include "menos/menos.h"

namespace menos {
namespace {

// A connected, non-Clifford block on qubits [first, first + count): no other
// engine undercuts dense on it.
void add_entangled_block(Circuit& circuit, QubitId first, std::uint32_t count) {
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        for (QubitId q = first; q < first + count; ++q) {
            circuit.append(Opcode::H, {q});
            circuit.append(Opcode::T, {q});
        }
        for (QubitId q = first; q + 1 < first + count; ++q) {
            circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
        }
    }
}

Circuit entangled(std::uint32_t num_qubits) {
    Circuit circuit(num_qubits, num_qubits);
    add_entangled_block(circuit, 0, num_qubits);
    return circuit;
}

constexpr std::uint64_t kAmplitude = sizeof(std::complex<double>);

TEST(PeakMemoryTest, ADenseStatevectorJobHoldsExactlyItsState) {
    const Result result =
        run(entangled(12), RunOptions{.shots = 0, .want_statevector = true, .seed = 1});
    ASSERT_EQ(result.plan().strategy, Strategy::DirectDense);
    // The state, moved into the result rather than copied (docs/adr/0025).
    EXPECT_EQ(result.peak_host_bytes(), kAmplitude << 12);
    EXPECT_EQ(result.peak_device_bytes(), 0U);
    EXPECT_LE(result.peak_host_bytes(), result.plan().estimated_state_bytes);
}

TEST(PeakMemoryTest, ADenseCountsJobAddsItsUniformsAndItsHistogram) {
    Circuit circuit = entangled(10);
    for (QubitId q = 0; q < 10; ++q) {
        circuit.append_measure(q, q);
    }
    const std::uint64_t shots = 1000;
    // The statevector request keeps the job on dense, which samples before
    // it hands the state over.
    const Result result =
        run(circuit, RunOptions{.shots = shots, .want_statevector = true, .seed = 3});
    ASSERT_EQ(result.plan().strategy, Strategy::DirectDense);
    // At the end of sampling: the state, one double per shot, and one node
    // (its value and four link words) and one key word per distinct outcome.
    const std::uint64_t histogram =
        result.counts().size() *
        (sizeof(std::pair<const BitString, std::uint64_t>) + 4 * sizeof(void*) + 8);
    EXPECT_EQ(result.peak_host_bytes(), (kAmplitude << 10) + shots * sizeof(double) + histogram);
    EXPECT_LE(result.peak_host_bytes(), result.plan().estimated_state_bytes);
}

TEST(PeakMemoryTest, AFactorizedStatevectorHoldsTheGlobalStateAndOneRegionAtATime) {
    // Two independent 6-qubit blocks.
    Circuit circuit(12, 0);
    add_entangled_block(circuit, 0, 6);
    add_entangled_block(circuit, 6, 6);
    const Result result = run(circuit, RunOptions{.shots = 0, .want_statevector = true, .seed = 1});
    ASSERT_EQ(result.plan().strategy, Strategy::Factorized);
    ASSERT_EQ(result.plan().regions.size(), 2U);
    // The global buffer, plus a region's state while it is multiplied in.
    EXPECT_EQ(result.peak_host_bytes(), (kAmplitude << 12) + (kAmplitude << 6));
    EXPECT_LE(result.peak_host_bytes(), result.plan().estimated_state_bytes);
}

TEST(PeakMemoryTest, StabilizerAndMpsJobsReportAPeakWithinTheirEstimates) {
    // A chain: MPS keeps a bond of 2. The same chain then twenty pairs across
    // the middle: a bond of up to 2^20 for MPS, so the stabilizer tableau wins.
    Circuit chain(40, 40);
    Circuit across(40, 40);
    for (QubitId q = 0; q < 40; ++q) {
        chain.append(Opcode::H, {q});
        across.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < 40; ++q) {
        chain.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
        across.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 0; q < 20; ++q) {
        across.append(Opcode::CX, {q, static_cast<QubitId>(q + 20)});
    }
    for (QubitId q = 0; q < 40; ++q) {
        chain.append_measure(q, q);
        across.append_measure(q, q);
    }
    for (const auto& [circuit, strategy] :
         {std::pair{chain, Strategy::DirectMPS}, std::pair{across, Strategy::DirectStabilizer}}) {
        const Result result = run(circuit, RunOptions{.shots = 100, .seed = 5});
        ASSERT_EQ(result.plan().strategy, strategy);
        EXPECT_GT(result.peak_host_bytes(), 0U);
        EXPECT_LE(result.peak_host_bytes(), result.plan().estimated_state_bytes);
        EXPECT_EQ(result.peak_device_bytes(), 0U);
    }
}

TEST(PeakMemoryTest, ExplainAppendsBothPeaksToTheActualLine) {
    const Result result =
        run(entangled(8), RunOptions{.shots = 0, .want_statevector = true, .seed = 1});
    const std::string text = result.explain();
    const std::size_t actual = text.find("actual     state_bytes=");
    ASSERT_NE(actual, std::string::npos);
    const std::string line = text.substr(actual, text.find('\n', actual) - actual);
    // After the pre-Phase-21 fields, so they stay where they were.
    EXPECT_LT(line.find("sample_s="), line.find("peak_host_bytes="));
    EXPECT_TRUE(line.ends_with(
        std::format(" peak_host_bytes={} peak_device_bytes=0", result.peak_host_bytes())))
        << line;
}

TEST(PeakMemoryTest, APlanWithNoCudaReportsNoDeviceFigures) {
    RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
    options.backend = Backend::Cpu;  // in a CUDA build too
    const Plan plan = preview(entangled(8), options);
    EXPECT_FALSE(plan.device_available_bytes.has_value());
    const std::string text = plan.explain();
    EXPECT_EQ(text.find("device_bytes="), std::string::npos);
    EXPECT_EQ(text.find("device_available_bytes="), std::string::npos);
}

}  // namespace
}  // namespace menos
