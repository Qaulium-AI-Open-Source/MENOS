// Phase 23 (docs/adr/0027 section 3.5, owner Q5): the planner reaches for the
// disk tier only when the in-memory dense state does not fit, or when the
// caller forces it; a forced tier never falls back; the report names the
// tier, its pass plan and the scratch figures; and a job that fits is planned
// exactly as before. Profiles are injected, so nothing here touches a disk.

#include "plan/planner.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/error.h"
#include "plan/hardware_profile.h"

namespace menos::plan {
namespace {

// Non-Clifford and entangled across the whole register (CX between the two
// halves, then along the chain), so neither the stabilizer nor the MPS
// engine can take the job and the dense algorithm is the only candidate.
Circuit entangled(std::uint32_t n, int layers) {
    Circuit circuit(n, n);
    for (int layer = 0; layer < layers; ++layer) {
        for (QubitId q = 0; q < n; ++q) {
            circuit.append(Opcode::H, {q});
            circuit.append(Opcode::RX, {q}, {0.3 + 0.1 * static_cast<double>(q)});
        }
        for (QubitId q = 0; q < n / 2; ++q) {
            circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + n / 2)});
        }
        for (QubitId q = 0; q + 1 < n; ++q) {
            circuit.append(Opcode::CX, {q, static_cast<QubitId>(q + 1)});
        }
    }
    for (QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

JobSpec counts_job(std::uint32_t n, std::optional<std::uint64_t> budget) {
    return JobSpec{
        .circuit = entangled(n, 3),
        .shots = 100,
        .want_statevector = false,
        .seed = 1,
        .memory_budget_bytes = budget,
        .expectation = std::nullopt,
        .scratch_directory = std::filesystem::path("/scratch"),
    };
}

// A profile with a scratch directory of a terabyte, half free, measured at
// step 1's figures; uncalibrated otherwise, so no seconds are computed.
HardwareProfile profile_with_scratch() {
    HardwareProfile hw;
    hw.total_memory_bytes = std::uint64_t{1} << 36;
    hw.scratch_capacity_bytes = std::uint64_t{1} << 40;
    hw.scratch_available_bytes = std::uint64_t{1} << 39;
    hw.scratch_write_bytes_per_second = 4.5e9;
    hw.scratch_read_bytes_per_second = 6.0e9;
    return hw;
}

HardwareProfile profile_without_scratch() {
    HardwareProfile hw;
    hw.total_memory_bytes = std::uint64_t{1} << 36;
    return hw;
}

HardwareProfile calibrated(HardwareProfile hw) {
    hw.work_units_per_second = 1.0e8;
    hw.memory_bandwidth_bytes_per_second = 1.0e10;
    return hw;
}

const CandidateSummary& dense_candidate(const Plan& plan) {
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy == Strategy::DirectDense) {
            return c;
        }
    }
    throw std::logic_error("no dense candidate");
}

// n = 14: 16 * 2^14 = 256 KiB of state; a budget of 64 KiB cannot hold it
// but can hold a window.
constexpr std::uint32_t kN = 14;
constexpr std::uint64_t kSmallBudget = std::uint64_t{64} << 10;

TEST(DiskTierSelectionTest, AStateOverTheBudgetWithCountsReachesTheDiskTier) {
    const Plan plan = plan_job(counts_job(kN, kSmallBudget), profile_with_scratch());
    ASSERT_EQ(plan.strategy, Strategy::DirectDense);
    ASSERT_EQ(plan.regions.size(), 1U);
    EXPECT_EQ(plan.regions[0].tier, Tier::Disk);
    EXPECT_EQ(plan.regions[0].backend, Backend::Cpu);
    const CandidateSummary& dense = dense_candidate(plan);
    EXPECT_TRUE(dense.feasible);
    EXPECT_EQ(dense.tier, Tier::Disk);
    EXPECT_GT(dense.tier_cost.passes, 0U);
    EXPECT_LT(dense.tier_cost.window_qubits, kN);
    EXPECT_LE(dense.tier_cost.chunk_qubits, dense.tier_cost.window_qubits);
    EXPECT_EQ(dense.tier_cost.file_bytes, std::uint64_t{16} << kN);
    EXPECT_LE(dense.estimated_peak_bytes, kSmallBudget);
    // The region hands the executor the plan the planner priced.
    EXPECT_EQ(plan.regions[0].tier_cost.chunk_qubits, dense.tier_cost.chunk_qubits);
    EXPECT_EQ(plan.regions[0].tier_cost.window_qubits, dense.tier_cost.window_qubits);
    EXPECT_EQ(plan.regions[0].estimated_state_bytes, dense.estimated_peak_bytes);
    EXPECT_EQ(plan.estimated_state_bytes, dense.estimated_peak_bytes);
    // The scratch figures the decision rested on.
    ASSERT_TRUE(plan.scratch_directory.has_value());
    EXPECT_EQ(*plan.scratch_directory, std::filesystem::path("/scratch"));
    EXPECT_EQ(plan.disk_available_bytes, std::optional<std::uint64_t>{std::uint64_t{1} << 39});
    EXPECT_FALSE(plan.requested_tier.has_value());

    const std::string report = plan.explain();
    EXPECT_NE(report.find(" tier=disk passes=" + std::to_string(dense.tier_cost.passes) +
                          " chunk_qubits=" + std::to_string(dense.tier_cost.chunk_qubits) +
                          " disk_bytes=" + std::to_string(dense.tier_cost.file_bytes)),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("engine=dense backend=cpu tier=disk"), std::string::npos) << report;
    EXPECT_NE(report.find(" disk_available_bytes=" + std::to_string(std::uint64_t{1} << 39) +
                          " scratch_directory=\"" + std::filesystem::path("/scratch").string() +
                          "\" disk_write_bandwidth=4.5e+09 disk_read_bandwidth=6e+09 (measured)"),
              std::string::npos)
        << report;
    EXPECT_EQ(report.find("(requested)"), std::string::npos) << report;
}

TEST(DiskTierSelectionTest, AStateThatFitsNeverGetsTheTierAndReportsAsBefore) {
    const JobSpec job = counts_job(kN, std::nullopt);
    const Plan with = plan_job(job, profile_with_scratch());
    const Plan without = plan_job(job, profile_without_scratch());
    EXPECT_EQ(with.explain(), without.explain());
    EXPECT_EQ(with.regions[0].tier, Tier::InMemory);
    EXPECT_EQ(dense_candidate(with).tier, Tier::InMemory);
    EXPECT_FALSE(with.scratch_directory.has_value());
    EXPECT_FALSE(with.disk_available_bytes.has_value());
    EXPECT_EQ(with.explain().find("tier="), std::string::npos);
    EXPECT_EQ(with.explain().find("disk"), std::string::npos);
}

TEST(DiskTierSelectionTest, AStatevectorRequestOverTheBudgetIsRefusedAsBefore) {
    JobSpec job = counts_job(kN, kSmallBudget);
    job.want_statevector = true;
    job.shots = 0;
    try {
        (void)plan_job(job, profile_with_scratch());
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("direct dense simulation requires"), std::string::npos) << what;
        EXPECT_EQ(what.find("disk"), std::string::npos) << what;
    }
}

TEST(DiskTierSelectionTest, TheReasonNamesTheDiskTierWhenItCannotHelpEither) {
    HardwareProfile hw = profile_with_scratch();
    hw.scratch_available_bytes = std::uint64_t{9} << 30;  // the reserve alone
    try {
        (void)plan_job(counts_job(kN, kSmallBudget), hw);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("direct dense simulation requires"), std::string::npos) << what;
        EXPECT_NE(what.find("the disk tier: the disk tier's scratch file of 262144 bytes plus "
                            "the reserve of " +
                            std::to_string(scratch_reserve_bytes(hw.scratch_capacity_bytes)) +
                            " bytes exceeds the 9663676416 bytes available in"),
                  std::string::npos)
            << what;
    }
    // No scratch directory at all.
    try {
        (void)plan_job(counts_job(kN, kSmallBudget), profile_without_scratch());
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("the disk tier has no scratch directory to use"),
                  std::string::npos)
            << e.what();
    }
}

TEST(DiskTierSelectionTest, TheReserveIsFivePercentOrEightGiB) {
    EXPECT_EQ(scratch_reserve_bytes(std::uint64_t{1} << 40), (std::uint64_t{1} << 40) / 20);
    EXPECT_EQ(scratch_reserve_bytes(std::uint64_t{100} << 30), std::uint64_t{8} << 30);
    EXPECT_EQ(scratch_reserve_bytes(0), std::uint64_t{8} << 30);
    // A file that leaves exactly the reserve is allowed; one byte less is not.
    HardwareProfile hw = profile_with_scratch();
    hw.scratch_capacity_bytes = std::uint64_t{100} << 30;
    hw.scratch_available_bytes = (std::uint64_t{8} << 30) + (std::uint64_t{16} << kN);
    EXPECT_EQ(plan_job(counts_job(kN, kSmallBudget), hw).regions[0].tier, Tier::Disk);
    hw.scratch_available_bytes -= 1;
    EXPECT_THROW((void)plan_job(counts_job(kN, kSmallBudget), hw), common::ResourceExceeded);
}

TEST(DiskTierSelectionTest, AForcedDiskTierOnAFittingJobUsesItAndNamesTheRequest) {
    JobSpec job = counts_job(kN, std::nullopt);
    job.tier = Tier::Disk;
    const Plan plan = plan_job(job, profile_with_scratch());
    ASSERT_EQ(plan.strategy, Strategy::DirectDense);
    EXPECT_EQ(plan.regions[0].tier, Tier::Disk);
    EXPECT_EQ(plan.requested_tier, std::optional<Tier>{Tier::Disk});
    const std::string report = plan.explain();
    EXPECT_NE(report.find("tier       disk (requested)\n"), std::string::npos) << report;
    bool refused_for_the_tier = false;
    for (const CandidateSummary& c : plan.candidates) {
        if (c.strategy != Strategy::DirectDense) {
            EXPECT_FALSE(c.feasible);
            // An engine's own refusal stands; the rest name the request.
            if (c.infeasible_reason.find("tier=disk requested: ") != std::string::npos) {
                EXPECT_NE(c.infeasible_reason.find(" has no disk tier"), std::string::npos)
                    << c.infeasible_reason;
                refused_for_the_tier = true;
            }
        }
    }
    EXPECT_TRUE(refused_for_the_tier);
    // With the whole state in memory the window holds it all: one pass.
    EXPECT_EQ(dense_candidate(plan).tier_cost.window_qubits, kN);
    EXPECT_EQ(dense_candidate(plan).tier_cost.passes, 1U);
}

TEST(DiskTierSelectionTest, AForcedTierNeverFallsBack) {
    JobSpec job = counts_job(kN, std::nullopt);
    job.tier = Tier::Disk;
    HardwareProfile no_space = profile_with_scratch();
    no_space.scratch_available_bytes = 0;
    try {
        (void)plan_job(job, no_space);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("tier=disk requested: "), std::string::npos)
            << e.what();
    }
    EXPECT_THROW((void)plan_job(job, profile_without_scratch()), common::ResourceExceeded);

    JobSpec statevector = job;
    statevector.want_statevector = true;
    EXPECT_THROW((void)plan_job(statevector, profile_with_scratch()), common::UnsupportedFeature);

    JobSpec cuda = job;
    cuda.backend = Backend::Cuda;
    EXPECT_THROW((void)plan_job(cuda, profile_with_scratch()), common::UnsupportedFeature);

    JobSpec offloaded = counts_job(kN, std::nullopt);
    offloaded.tier = Tier::HostOffloaded;
    EXPECT_THROW((void)plan_job(offloaded, profile_with_scratch()), common::UnsupportedFeature);
}

TEST(DiskTierSelectionTest, AForcedInMemoryTierRefusesAStateOverTheBudgetAsBefore) {
    JobSpec job = counts_job(kN, kSmallBudget);
    job.tier = Tier::InMemory;
    try {
        (void)plan_job(job, profile_with_scratch());
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_EQ(std::string(e.what()).find("disk"), std::string::npos) << e.what();
    }
    const Plan fits = plan_job(counts_job(kN, std::nullopt), profile_with_scratch());
    job.memory_budget_bytes = std::nullopt;
    const Plan forced = plan_job(job, profile_with_scratch());
    EXPECT_EQ(forced.regions[0].tier, Tier::InMemory);
    EXPECT_NE(forced.explain().find("tier       in_memory (requested)\n"), std::string::npos);
    EXPECT_EQ(forced.explain().size(),
              fits.explain().size() + std::string("tier       in_memory (requested)\n").size());
}

TEST(DiskTierSelectionTest, TheTimeBudgetUsesTheTiersSeconds) {
    const HardwareProfile hw = calibrated(profile_with_scratch());
    JobSpec job = counts_job(kN, kSmallBudget);
    job.time_budget_seconds = 1.0e9;
    const Plan plan = plan_job(job, hw);
    const CandidateSummary& dense = dense_candidate(plan);
    ASSERT_EQ(dense.tier, Tier::Disk);
    const double expected = static_cast<double>(dense.estimated_work_units) / 1.0e8 +
                            static_cast<double>(dense.tier_cost.link_bytes) / 4.5e9;
    EXPECT_NEAR(dense.estimated_seconds, expected, 1e-9 * expected);
    EXPECT_GT(dense.tier_cost.link_bytes,
              dense.tier_cost.file_bytes * (1 + 2 * dense.tier_cost.passes));
    // A budget below it refuses the job with the tier's figure.
    job.time_budget_seconds = expected / 2.0;
    EXPECT_THROW((void)plan_job(job, hw), common::ResourceExceeded);
    // Unmeasured bandwidth under a time budget is a refusal, never a guess;
    // without a budget the figure is simply absent.
    HardwareProfile unmeasured = hw;
    unmeasured.scratch_write_bytes_per_second = 0.0;
    unmeasured.scratch_read_bytes_per_second = 0.0;
    job.time_budget_seconds = 1.0e9;
    try {
        (void)plan_job(job, unmeasured);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("bandwidth is unmeasured"), std::string::npos)
            << e.what();
    }
    job.time_budget_seconds = std::nullopt;
    const Plan absent = plan_job(job, unmeasured);
    EXPECT_EQ(dense_candidate(absent).tier, Tier::Disk);
    EXPECT_LT(dense_candidate(absent).estimated_seconds, 0.0);
    EXPECT_NE(absent.explain().find(" disk_bandwidth=unmeasured"), std::string::npos)
        << absent.explain();
}

TEST(DiskTierSelectionTest, TheReachTestSaysYesExactlyWhenTheTierIsNotRuledOut) {
    const HardwareProfile hw = profile_with_scratch();
    EXPECT_TRUE(may_reach_disk_tier(counts_job(kN, kSmallBudget), hw));
    EXPECT_FALSE(may_reach_disk_tier(counts_job(kN, std::nullopt), hw));
    JobSpec forced = counts_job(kN, std::nullopt);
    forced.tier = Tier::Disk;
    EXPECT_TRUE(may_reach_disk_tier(forced, hw));
    JobSpec in_memory = counts_job(kN, kSmallBudget);
    in_memory.tier = Tier::InMemory;
    EXPECT_FALSE(may_reach_disk_tier(in_memory, hw));
    JobSpec statevector = counts_job(kN, kSmallBudget);
    statevector.want_statevector = true;
    EXPECT_FALSE(may_reach_disk_tier(statevector, hw));
    JobSpec cuda = counts_job(kN, kSmallBudget);
    cuda.backend = Backend::Cuda;
    EXPECT_FALSE(may_reach_disk_tier(cuda, hw));
    // The budget that the state alone meets but the outputs push over: the
    // planner reaches for the tier, and so does the test.
    JobSpec tight = counts_job(kN, std::uint64_t{16} << kN);
    EXPECT_TRUE(may_reach_disk_tier(tight, hw));
    EXPECT_EQ(plan_job(tight, hw).regions[0].tier, Tier::Disk);
}

TEST(DiskTierSelectionTest, AnObservedFloorOnTheDiskTierDoesNotRaiseTheInMemoryOne) {
    const HardwareProfile hw = calibrated(profile_with_scratch());
    JobSpec job = counts_job(kN, std::nullopt);
    job.time_budget_seconds = 10.0;
    RuntimeObservations observations;
    observations.observe(Engine::Dense, Backend::Cpu, Tier::Disk, 1.0e9);
    const Plan plan = plan_job(job, hw, observations);
    EXPECT_EQ(plan.regions[0].tier, Tier::InMemory);
    EXPECT_TRUE(dense_candidate(plan).feasible);
    // And the reverse: a floor on the in-memory engine leaves a disk-tier
    // job's figure alone, while a floor on the tier itself refuses it.
    JobSpec over = counts_job(kN, kSmallBudget);
    over.time_budget_seconds = 10.0;
    RuntimeObservations in_memory;
    in_memory.observe(Engine::Dense, Backend::Cpu, 1.0e9);
    EXPECT_EQ(plan_job(over, hw, in_memory).regions[0].tier, Tier::Disk);
    EXPECT_THROW((void)plan_job(over, hw, observations), common::ResourceExceeded);
}

}  // namespace
}  // namespace menos::plan
