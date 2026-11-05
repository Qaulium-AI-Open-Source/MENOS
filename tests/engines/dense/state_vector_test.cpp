#include "engines/dense/state_vector.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "common/error.h"

namespace menos::engines::dense {
namespace {

TEST(StateVectorTest, SizedCorrectly) {
    const StateVector state(3, /*budget_bytes=*/1'000'000);
    EXPECT_EQ(state.num_qubits(), 3);
    EXPECT_EQ(state.size_bytes(), 8U * sizeof(std::complex<double>));
    EXPECT_EQ(state.amplitudes().size(), 8U);
}

// The amplitudes leave by move: the released vector owns the very buffer the
// state held, so a requested statevector costs no second copy (docs/adr/0025).
TEST(StateVectorTest, ReleaseHandsOverTheBufferWithoutCopying) {
    StateVector state(5, /*budget_bytes=*/1'000'000);
    initialize(state, 0.0);
    const std::complex<double>* buffer = state.amplitudes().data();
    const std::vector<std::complex<double>> released = std::move(state).release();
    EXPECT_EQ(released.data(), buffer);
    ASSERT_EQ(released.size(), 32U);
    EXPECT_EQ(released[0], std::complex<double>(1.0, 0.0));
}

TEST(StateVectorTest, ThrowsResourceExceededBeforeAllocatingWhenBudgetTooSmall) {
    EXPECT_THROW(StateVector(40, /*budget_bytes=*/1024), common::ResourceExceeded);
}

TEST(StateVectorTest, ZeroQubitsIsOneAmplitude) {
    const StateVector state(0, /*budget_bytes=*/1024);
    EXPECT_EQ(state.amplitudes().size(), 1U);
}

TEST(InitializeTest, DefaultsToBasisZeroWithNoPhase) {
    StateVector state(2, /*budget_bytes=*/1'000'000);
    initialize(state, 0.0);
    const auto amps = state.amplitudes();
    EXPECT_EQ(amps[0], std::complex<double>(1.0, 0.0));
    for (std::size_t i = 1; i < amps.size(); ++i) {
        EXPECT_EQ(amps[i], std::complex<double>(0.0, 0.0));
    }
}

TEST(InitializeTest, AppliesGlobalPhase) {
    StateVector state(1, /*budget_bytes=*/1'000'000);
    const double phase = std::numbers::pi / 3.0;
    initialize(state, phase);
    const auto amps = state.amplitudes();
    EXPECT_NEAR(amps[0].real(), std::cos(phase), 1e-12);
    EXPECT_NEAR(amps[0].imag(), std::sin(phase), 1e-12);
    EXPECT_EQ(amps[1], std::complex<double>(0.0, 0.0));
}

}  // namespace
}  // namespace menos::engines::dense
