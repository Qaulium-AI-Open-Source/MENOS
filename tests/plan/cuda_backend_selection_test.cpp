#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/error.h"
#include "engines/dense_cuda/cuda_dense_engine.h"
#include "plan/hardware_profile.h"
#include "plan/planner.h"

namespace menos::plan {
namespace {

// Backend selection is the two-level rule from docs/adr/0014: the ALGORITHM is
// chosen by work units exactly as before, and only then is the BACKEND chosen
// by calibrated wall-clock seconds, with ties going to the CPU.
//
// Most of these use a SYNTHETIC HardwareProfile rather than the calibrated one.
// That is deliberate: a test built on this machine's measured rates would
// assert how fast this particular laptop is, and would change meaning on other
// hardware. With the rates pinned, the decision the planner makes is a pure
// function of the numbers and means the same thing everywhere. The device
// itself still has to be real -- the CUDA engine is only registered when one is
// present -- so these also confirm the engine is genuinely reachable.

class CudaBackendSelectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!engines::dense_cuda::cuda_device_available()) {
            GTEST_SKIP() << "no CUDA device available -- backend-selection coverage did NOT run";
        }
    }

    static JobSpec dense_job(std::uint32_t num_qubits) {
        // Enough gates that the work term is not trivially zero, and H/CX so
        // the circuit is not Clifford-only in a way that would hand the job to
        // the stabilizer engine instead of dense.
        Circuit circuit(num_qubits, 0);
        for (std::uint32_t layer = 0; layer < 4; ++layer) {
            for (std::uint32_t q = 0; q < num_qubits; ++q) {
                circuit.append(Opcode::H, {static_cast<QubitId>(q)});
            }
            circuit.append(Opcode::T, {0});  // non-Clifford: keeps dense the winner
            for (std::uint32_t q = layer % 2; q + 1 < num_qubits; q += 2) {
                circuit.append(Opcode::CX, {static_cast<QubitId>(q), static_cast<QubitId>(q + 1)});
            }
        }
        return JobSpec{
            .circuit = circuit,
            .shots = 0,
            .want_statevector = true,
            .seed = 0,
            .memory_budget_bytes = std::uint64_t{1} << 34,
            .expectation = std::nullopt,
        };
    }

    // A profile where both backends are calibrated and the device is
    // enormously faster per work unit, with negligible launch overhead.
    static HardwareProfile fast_device_profile() {
        HardwareProfile hw;
        hw.total_memory_bytes = std::uint64_t{1} << 36;
        hw.memory_bandwidth_bytes_per_second = 1.0e10;
        hw.work_units_per_second = 1.0e7;
        hw.cuda_present = true;
        hw.cuda_device_total_bytes = std::uint64_t{1} << 32;
        hw.cuda_device_available_bytes = std::uint64_t{1} << 32;
        hw.cuda_host_to_device_bytes_per_second = 1.0e10;
        hw.cuda_device_to_host_bytes_per_second = 1.0e10;
        hw.cuda_kernel_launch_seconds = 1.0e-9;
        hw.cuda_device_work_units_per_second = 1.0e10;
        return hw;
    }
};

TEST_F(CudaBackendSelectionTest, AFasterDeviceWinsTheDenseRegion) {
    const Plan plan = plan_job(dense_job(12), fast_device_profile());
    ASSERT_EQ(plan.strategy, Strategy::DirectDense);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].engine, Engine::Dense) << "the ALGORITHM must not change";
    EXPECT_EQ(plan.regions[0].backend, Backend::Cuda);
}

// The launch term is the whole reason a crossover exists. With a launch cost
// large enough to dominate, the device must lose even though its per-work-unit
// rate is a thousand times better -- a cost model without this term would
// wrongly claim the GPU always wins.
TEST_F(CudaBackendSelectionTest, LaunchOverheadAloneCanMakeTheDeviceLose) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_kernel_launch_seconds = 1.0;  // one second per launch
    const Plan plan = plan_job(dense_job(12), hw);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
}

TEST_F(CudaBackendSelectionTest, TiesGoToTheCpu) {
    HardwareProfile hw = fast_device_profile();
    // Make the device's total predicted time exactly equal the CPU's by giving
    // it the same work rate and removing every other term.
    hw.cuda_device_work_units_per_second = hw.work_units_per_second;
    hw.cuda_kernel_launch_seconds = 0.0;
    hw.cuda_host_to_device_bytes_per_second = 1.0e300;
    hw.cuda_device_to_host_bytes_per_second = 1.0e300;
    // The CPU figure is max(work, traffic); make the traffic term irrelevant so
    // both sides reduce to the same work term.
    hw.memory_bandwidth_bytes_per_second = 1.0e300;

    const Plan plan = plan_job(dense_job(10), hw);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu)
        << "equal estimates must not move work to the GPU";
}

// Device memory is a budget separate from host memory. A region that fits
// comfortably in host memory must still be refused on the device when it does
// not fit there -- and the plan must fall back to the CPU rather than fail.
TEST_F(CudaBackendSelectionTest, ADeviceTooSmallForTheStateKeepsTheRegionOnTheCpu) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_device_available_bytes = 1024;  // far too small for any real state
    const Plan plan = plan_job(dense_job(12), hw);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense) << "the plan must still succeed on the CPU";
}

// An uncalibrated profile cannot compare backends, so it must not guess.
TEST_F(CudaBackendSelectionTest, AnUncalibratedProfileNeverSelectsTheDevice) {
    HardwareProfile hw;
    hw.total_memory_bytes = std::uint64_t{1} << 36;
    const Plan plan = plan_job(dense_job(12), hw);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
}

// The real crossover, against this machine's own calibrated profile rather than
// synthetic numbers. This is the end-to-end claim: the cost model, fed only by
// measurements, reproduces the fact that small circuits belong on the CPU and
// large ones on the GPU. The qubit counts are far apart on purpose so the
// assertion does not depend on exactly where the crossover falls.
//
// The large case (owner, 2026-11-12, docs/adr/0028 section 23) is a 24-qubit
// expectation. On the development machine's RTX 4060 the crossover benchmark
// measured, for a statevector, the CPU faster at 20 qubits (29 ms against
// 40 ms), a tie at 22 and the GPU 1.23x faster at 24: a margin inside the
// time model's error, so a statevector case asserted either what was not
// true there or what the model cannot resolve. An expectation copies no state
// back, and the GPU is measured 1.64x faster at 24. Z on every qubit keeps
// the whole circuit in the light cone. The check is on the dense candidate,
// the one the CPU/CUDA comparison decides; which engine then wins the plan is
// another question.
TEST_F(CudaBackendSelectionTest,
       TheCalibratedProfileKeepsTinyCircuitsOnTheCpuAndLargeOnesOnTheGpu) {
    const HardwareProfile& hw = calibrated_hardware_profile();
    if (!hw.cuda_present || hw.cuda_device_work_units_per_second <= 0.0) {
        GTEST_SKIP() << "CUDA calibration unavailable in this configuration";
    }

    const Plan small = plan_job(dense_job(3), hw);
    ASSERT_EQ(small.regions.size(), 1U);
    EXPECT_EQ(small.regions[0].backend, Backend::Cpu)
        << "a 3-qubit circuit is dominated by launch overhead and belongs on the CPU";

    constexpr std::uint32_t kLarge = 24;
    JobSpec large_job = dense_job(kLarge);
    large_job.want_statevector = false;
    PauliString all_z{.num_qubits = kLarge, .x_words = {0}, .z_words = {(1ULL << kLarge) - 1}};
    large_job.expectation = Observable{.terms = {{all_z, {1.0, 0.0}}}};
    const Plan large = plan_job(large_job, hw);
    const auto dense = std::find_if(
        large.candidates.begin(), large.candidates.end(), [](const CandidateSummary& c) {
            return c.strategy == Strategy::DirectDense;
        });
    ASSERT_NE(dense, large.candidates.end());
    ASSERT_TRUE(dense->feasible) << dense->infeasible_reason;
    EXPECT_EQ(dense->backend, Backend::Cuda)
        << "a 24-qubit dense expectation should be faster on this GPU";
}

// ---------------------------------------------------------------------------
// Phase 11: an observed overrun steers the backend choice (docs/adr/0015).
//
// Backend selection used to run before an observed floor was applied, so after
// a GPU run overran its budget the model still preferred the GPU, the planner
// picked it again, and only then did the time-budget test find it infeasible --
// refusing a job the CPU was estimated to fit. These pin the fixed behaviour on
// the exact case that was reproduced, using the same synthetic profile as the
// rest of this file so the result means the same on every machine.
// ---------------------------------------------------------------------------

constexpr double kBudgetSeconds = 1000.0;
constexpr double kOverrunSeconds = 2000.0;  // strictly above the budget, as a crossing is

const CandidateSummary& winner_of(const Plan& plan) {
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.is_winner) {
            return candidate;
        }
    }
    throw std::logic_error("plan has no winning candidate");
}

// The exact reproduced case: initial DirectDense on CUDA, the CUDA run overran,
// the CPU was never tried and is estimated to fit -- so the replan must pick
// DirectDense on the CPU, not throw ResourceExceeded.
TEST_F(CudaBackendSelectionTest, ACudaOverrunSteersTheDenseRegionOntoTheCpu) {
    JobSpec job = dense_job(12);
    job.time_budget_seconds = kBudgetSeconds;
    const HardwareProfile hw = fast_device_profile();

    const Plan initial = plan_job(job, hw);
    ASSERT_EQ(initial.strategy, Strategy::DirectDense);
    ASSERT_EQ(initial.regions[0].backend, Backend::Cuda) << "the scenario starts on the GPU";

    // The CPU figure the replan should land on, taken from the same job planned
    // with no device at all, so this test does not restate the formula.
    HardwareProfile no_device = hw;
    no_device.cuda_present = false;
    const double cpu_seconds = winner_of(plan_job(job, no_device)).estimated_seconds;
    ASSERT_LT(cpu_seconds, kBudgetSeconds) << "precondition: the CPU fits the budget";
    ASSERT_GT(cpu_seconds, winner_of(initial).estimated_seconds)
        << "precondition: the model preferred the GPU, which is what makes this case";

    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cuda, kOverrunSeconds);

    const Plan replanned = plan_job(job, hw, observations);  // must not throw
    EXPECT_EQ(replanned.strategy, Strategy::DirectDense);
    ASSERT_EQ(replanned.regions.size(), 1U);
    EXPECT_EQ(replanned.regions[0].engine, Engine::Dense) << "the ALGORITHM must not change";
    EXPECT_EQ(replanned.regions[0].backend, Backend::Cpu);
    EXPECT_TRUE(winner_of(replanned).feasible);
    EXPECT_DOUBLE_EQ(winner_of(replanned).estimated_seconds, cpu_seconds);
    EXPECT_LE(winner_of(replanned).estimated_seconds, kBudgetSeconds);
}

// The converse: with an over-budget floor on BOTH backends there is nothing left
// to steer to, so the strict refusal still happens.
TEST_F(CudaBackendSelectionTest, AnOverrunOnBothBackendsIsStillRefused) {
    JobSpec job = dense_job(12);  // statevector: only dense can serve it
    job.time_budget_seconds = kBudgetSeconds;
    const HardwareProfile hw = fast_device_profile();

    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cuda, kOverrunSeconds);
    observations.observe(Engine::Dense, Backend::Cpu, kOverrunSeconds);

    try {
        static_cast<void>(plan_job(job, hw, observations));
        FAIL() << "both backends overran, so no plan exists and planning must throw";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("observed runtime"), std::string::npos)
            << "the refusal must say the figure was observed: " << e.what();
    }
}

// Steering is not a promise to find a fitting plan. A CPU that the MODEL says is
// too slow stays refused even after the GPU overran: the observation moved the
// choice, it did not make the CPU faster.
TEST_F(CudaBackendSelectionTest, ACpuTheModelSaysIsTooSlowIsStillRefusedAfterACudaOverrun) {
    JobSpec job = dense_job(12);
    const HardwareProfile hw = fast_device_profile();
    HardwareProfile no_device = hw;
    no_device.cuda_present = false;
    const double cpu_seconds = winner_of(plan_job(job, no_device)).estimated_seconds;

    job.time_budget_seconds = cpu_seconds / 2.0;  // GPU fits, CPU does not
    ASSERT_EQ(plan_job(job, hw).regions[0].backend, Backend::Cuda);

    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cuda, kOverrunSeconds);
    EXPECT_THROW(static_cast<void>(plan_job(job, hw, observations)), common::ResourceExceeded);
}

// The mirror image: a job that started on the CPU and overran there is steered
// to the GPU when the GPU fits.
TEST_F(CudaBackendSelectionTest, ACpuOverrunSteersTheDenseRegionOntoTheGpu) {
    JobSpec job = dense_job(12);
    job.time_budget_seconds = kBudgetSeconds;
    HardwareProfile hw = fast_device_profile();
    hw.cuda_kernel_launch_seconds = 1.0e-2;  // ~0.8 s of launches: the CPU wins first

    const Plan initial = plan_job(job, hw);
    ASSERT_EQ(initial.regions[0].backend, Backend::Cpu) << "the scenario starts on the CPU";

    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cpu, kOverrunSeconds);

    const Plan replanned = plan_job(job, hw, observations);
    EXPECT_EQ(replanned.strategy, Strategy::DirectDense);
    EXPECT_EQ(replanned.regions[0].backend, Backend::Cuda);
    EXPECT_LE(winner_of(replanned).estimated_seconds, kBudgetSeconds);
}

// An observation only ever RAISES a figure. One below the model must not move
// the choice -- the same max() rule the time-budget test uses, applied here.
TEST_F(CudaBackendSelectionTest, AFloorBelowTheModelLeavesTheBackendChoiceAlone) {
    JobSpec job = dense_job(12);
    job.time_budget_seconds = kBudgetSeconds;
    const HardwareProfile hw = fast_device_profile();

    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cuda, 1.0e-12);

    const Plan replanned = plan_job(job, hw, observations);
    EXPECT_EQ(replanned.regions[0].backend, Backend::Cuda);
    EXPECT_DOUBLE_EQ(winner_of(replanned).estimated_seconds,
                     winner_of(plan_job(job, hw)).estimated_seconds);
}

// With no observations the comparison is exactly the Phase 10 one: this is what
// keeps every first planning, and every caller that never replans, unchanged.
TEST_F(CudaBackendSelectionTest, NoObservationsLeavesBackendSelectionExactlyAsItWas) {
    JobSpec job = dense_job(12);
    const HardwareProfile hw = fast_device_profile();
    const Plan without = plan_job(job, hw);
    const Plan with_empty = plan_job(job, hw, RuntimeObservations{});
    EXPECT_EQ(with_empty.regions[0].backend, without.regions[0].backend);
    EXPECT_DOUBLE_EQ(winner_of(with_empty).estimated_seconds, winner_of(without).estimated_seconds);
}

// ---------------------------------------------------------------------------
// Phase 17: a forced backend (JobSpec::backend, docs/adr/0021 section 2).
// ---------------------------------------------------------------------------

TEST_F(CudaBackendSelectionTest, ForcedCudaRunsOnTheDeviceEvenWhereTheCpuIsFaster) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_kernel_launch_seconds = 1.0;  // the model now prefers the CPU
    JobSpec job = dense_job(12);
    ASSERT_EQ(plan_job(job, hw).regions[0].backend, Backend::Cpu) << "precondition";
    job.backend = Backend::Cuda;
    const Plan plan = plan_job(job, hw);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].engine, Engine::Dense);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cuda);
    EXPECT_EQ(plan.requested_backend, Backend::Cuda);
    EXPECT_NE(plan.explain().find("backend    cuda (requested)\n"), std::string::npos)
        << plan.explain();
}

TEST_F(CudaBackendSelectionTest, ForcedCpuStaysOnTheCpuWhereTheDeviceIsFaster) {
    const HardwareProfile hw = fast_device_profile();
    JobSpec job = dense_job(12);
    ASSERT_EQ(plan_job(job, hw).regions[0].backend, Backend::Cuda) << "precondition";
    job.backend = Backend::Cpu;
    const Plan plan = plan_job(job, hw);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
    EXPECT_NE(plan.explain().find("backend    cpu (requested)\n"), std::string::npos);
}

TEST_F(CudaBackendSelectionTest, ForcedCudaOnACliffordCircuitIsPlannedAsDenseOnTheDevice) {
    // Without a request a CPU-only engine (not dense) wins this Clifford
    // circuit. Forced CUDA leaves only the dense candidate, on the device.
    Circuit circuit(10, 10);
    for (QubitId q = 0; q < 10; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < 10; ++q) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    for (QubitId q = 0; q < 10; ++q) {
        circuit.append_measure(q, q);
    }
    JobSpec job = dense_job(10);
    job.circuit = circuit;
    job.want_statevector = false;
    job.shots = 16;
    const HardwareProfile hw = fast_device_profile();
    ASSERT_NE(plan_job(job, hw).strategy, Strategy::DirectDense) << "precondition";
    job.backend = Backend::Cuda;
    const Plan plan = plan_job(job, hw);
    EXPECT_EQ(plan.strategy, Strategy::DirectDense);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cuda);
    // The other candidates are listed as infeasible with the reason.
    EXPECT_NE(plan.explain().find("backend=cuda requested: direct_stabilizer has no CUDA backend"),
              std::string::npos)
        << plan.explain();
}

TEST_F(CudaBackendSelectionTest, ForcedCudaThatDoesNotFitTheDeviceIsResourceExceeded) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_device_available_bytes = 1024;
    JobSpec job = dense_job(12);
    job.backend = Backend::Cuda;
    EXPECT_THROW(static_cast<void>(plan_job(job, hw)), common::ResourceExceeded);
}

TEST_F(CudaBackendSelectionTest, ForcedCudaWithoutADeviceIsUnsupported) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_present = false;
    JobSpec job = dense_job(12);
    job.backend = Backend::Cuda;
    EXPECT_THROW(static_cast<void>(plan_job(job, hw)), common::UnsupportedFeature);
}

// Owner Q1: an overrun on a forced backend is refused, never steered to the
// other one. The same scenario unforced is steered to the CPU
// (ACudaOverrunSteersTheDenseRegionOntoTheCpu above).
TEST_F(CudaBackendSelectionTest, AForcedCudaOverrunIsRefusedNotMovedToTheCpu) {
    JobSpec job = dense_job(12);
    job.time_budget_seconds = kBudgetSeconds;
    job.backend = Backend::Cuda;
    const HardwareProfile hw = fast_device_profile();
    ASSERT_EQ(plan_job(job, hw).regions[0].backend, Backend::Cuda) << "precondition";

    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cuda, kOverrunSeconds);
    EXPECT_THROW(static_cast<void>(plan_job(job, hw, observations)), common::ResourceExceeded);
}

// ---------------------------------------------------------------------------
// Phase 17 item 4b: per-region backends of a Factorized plan (docs/adr/0021
// section 10). Each dense region is decided by the same comparison as the
// direct dense candidate.
// ---------------------------------------------------------------------------

// One scrambling block, as in benchmarks/cuda_factorized_backend_benchmark.cpp
// (H and T on every qubit, CX on a random pairing per layer, fixed seed):
// connected and non-Clifford, so its region is dense.
void append_scrambling_block(Circuit& circuit, QubitId offset, std::uint32_t n) {
    std::mt19937 rng(1700 + n);
    std::vector<std::uint32_t> order(n);
    for (std::uint32_t layer = 0; layer < 6; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::H, {static_cast<QubitId>(offset + q)});
            circuit.append(Opcode::T, {static_cast<QubitId>(offset + q)});
        }
        for (std::uint32_t q = 0; q < n; ++q) {
            order[q] = q;
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (std::uint32_t q = 0; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX,
                           {static_cast<QubitId>(offset + order[q]),
                            static_cast<QubitId>(offset + order[q + 1])});
        }
    }
}

// A 6-qubit block (region 0) and a 12-qubit block (region 1), every qubit
// measured: a Factorized plan of two dense regions.
JobSpec two_block_job() {
    Circuit circuit(18, 18);
    append_scrambling_block(circuit, 0, 6);
    append_scrambling_block(circuit, 6, 12);
    for (QubitId q = 0; q < 18; ++q) {
        circuit.append_measure(q, q);
    }
    return JobSpec{.circuit = circuit,
                   .shots = 100,
                   .want_statevector = false,
                   .seed = 0,
                   .memory_budget_bytes = std::uint64_t{1} << 34,
                   .expectation = std::nullopt};
}

const CandidateSummary& factorized_candidate(const Plan& plan) {
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.strategy == Strategy::Factorized) {
            return candidate;
        }
    }
    throw std::logic_error("no factorized candidate");
}

class FactorizedRegionBackendTest : public CudaBackendSelectionTest {
protected:
    // Preconditions every test below relies on, checked once per test.
    static void expect_two_dense_regions(const Plan& plan) {
        ASSERT_EQ(plan.strategy, Strategy::Factorized);
        ASSERT_EQ(plan.regions.size(), 2U);
        ASSERT_EQ(plan.regions[0].engine, Engine::Dense);
        ASSERT_EQ(plan.regions[1].engine, Engine::Dense);
        ASSERT_EQ(plan.regions[0].local_to_global.size(), 6U);
        ASSERT_EQ(plan.regions[1].local_to_global.size(), 12U);
    }
};

TEST_F(FactorizedRegionBackendTest, AFasterDeviceTakesEveryDenseRegion) {
    const Plan plan = plan_job(two_block_job(), fast_device_profile());
    expect_two_dense_regions(plan);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cuda);
    EXPECT_EQ(plan.regions[1].backend, Backend::Cuda);
    EXPECT_NE(plan.explain().find("engine=dense backend=cuda"), std::string::npos);
}

TEST_F(FactorizedRegionBackendTest, ExpensiveLaunchesKeepThePlanExactlyAsWithoutADevice) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_kernel_launch_seconds = 1.0;
    HardwareProfile no_device = hw;
    no_device.cuda_present = false;
    const Plan with_device = plan_job(two_block_job(), hw);
    expect_two_dense_regions(with_device);
    EXPECT_EQ(with_device.regions[0].backend, Backend::Cpu);
    EXPECT_EQ(with_device.regions[1].backend, Backend::Cpu);
    EXPECT_EQ(with_device.explain(), plan_job(two_block_job(), no_device).explain());
}

TEST_F(FactorizedRegionBackendTest, EachRegionIsDecidedOnItsOwn) {
    // Device memory for the 6-qubit state (1 KiB) but not the 12-qubit one
    // (64 KiB): the small region moves, the large one cannot.
    HardwareProfile hw = fast_device_profile();
    hw.cuda_device_available_bytes = 4096;
    const Plan plan = plan_job(two_block_job(), hw);
    expect_two_dense_regions(plan);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cuda);
    EXPECT_EQ(plan.regions[1].backend, Backend::Cpu);
}

TEST_F(FactorizedRegionBackendTest, ForcedCpuKeepsEveryRegionOnTheCpu) {
    JobSpec job = two_block_job();
    job.backend = Backend::Cpu;
    const Plan plan = plan_job(job, fast_device_profile());
    expect_two_dense_regions(plan);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
    EXPECT_EQ(plan.regions[1].backend, Backend::Cpu);
}

TEST_F(FactorizedRegionBackendTest, ForcedCudaPutsEveryDenseRegionOnTheDevice) {
    HardwareProfile hw = fast_device_profile();
    hw.cuda_kernel_launch_seconds = 1.0;  // the model alone would keep both on the CPU
    JobSpec job = two_block_job();
    job.backend = Backend::Cuda;
    const Plan plan = plan_job(job, hw);
    expect_two_dense_regions(plan);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cuda);
    EXPECT_EQ(plan.regions[1].backend, Backend::Cuda);
}

TEST_F(FactorizedRegionBackendTest, ForcedCudaWithANonDenseRegionRefusesTheFactorizedPlan) {
    // A Clifford chain runs on a CPU-only engine (MPS or stabilizer, whichever
    // is cheaper), so the Factorized candidate is infeasible and says why.
    Circuit circuit(18, 18);
    for (QubitId q = 0; q < 6; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (QubitId q = 0; q + 1 < 6; ++q) {
        circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
    }
    append_scrambling_block(circuit, 6, 12);
    for (QubitId q = 0; q < 18; ++q) {
        circuit.append_measure(q, q);
    }
    JobSpec job = two_block_job();
    job.circuit = circuit;
    job.backend = Backend::Cuda;
    const Plan plan = plan_job(job, fast_device_profile());
    const CandidateSummary& factorized = factorized_candidate(plan);
    EXPECT_FALSE(factorized.feasible);
    const std::string& reason = factorized.infeasible_reason;
    EXPECT_EQ(reason.rfind("backend=cuda requested: region 0 runs on ", 0), 0U) << reason;
    EXPECT_NE(reason.find(", which has no CUDA backend"), std::string::npos) << reason;
    EXPECT_EQ(reason.find("dense"), std::string::npos) << reason;
    EXPECT_NE(plan.strategy, Strategy::Factorized);
}

TEST_F(FactorizedRegionBackendTest, TheTimeBudgetIsJudgedAgainstThePerRegionBackends) {
    // A budget between the mixed and the all-CPU estimates: the Factorized
    // candidate fits only because its regions run on the device.
    JobSpec job = two_block_job();
    job.time_budget_seconds = 1.0e9;  // non-binding, to read both estimates
    const HardwareProfile hw = fast_device_profile();
    HardwareProfile no_device = hw;
    no_device.cuda_present = false;
    const double mixed = factorized_candidate(plan_job(job, hw)).estimated_seconds;
    const double all_cpu = factorized_candidate(plan_job(job, no_device)).estimated_seconds;
    ASSERT_GT(mixed, 0.0);
    ASSERT_LT(mixed, all_cpu) << "precondition: the device is faster here";

    job.time_budget_seconds = (mixed + all_cpu) / 2.0;
    EXPECT_TRUE(factorized_candidate(plan_job(job, hw)).feasible);
    // Without the device nothing fits that budget, so the job is refused.
    EXPECT_THROW(static_cast<void>(plan_job(job, no_device)), common::ResourceExceeded);
}

}  // namespace
}  // namespace menos::plan
