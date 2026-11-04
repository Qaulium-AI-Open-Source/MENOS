// Phase 23 (docs/adr/0027): RunOptions::tier and RunOptions::scratch_directory
// through the public API, in every build. The disk tier is reached the public
// way -- a memory budget the dense state does not fit -- and forced; its
// counts and expectation values are bitwise the in-memory run's; the report
// names the tier; the scratch file is gone afterwards; a forced tier never
// falls back.

#include <gtest/gtest.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <bit>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

#include "common/error.h"
#include "menos/menos.h"

namespace menos {
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

Circuit bell() {
    Circuit circuit(2, 2);
    circuit.append(Opcode::H, {0});
    circuit.append(Opcode::CX, {0, 1});
    circuit.append_measure(0, 0);
    circuit.append_measure(1, 1);
    return circuit;
}

Observable three_terms(std::uint32_t n) {
    const std::uint64_t all = (std::uint64_t{1} << n) - 1;
    Observable observable;
    observable.terms.emplace_back(PauliString{n, {0}, {all}}, std::complex<double>{1.0, 0.0});
    observable.terms.emplace_back(PauliString{n, {all}, {0}}, std::complex<double>{0.5, 0.0});
    observable.terms.emplace_back(PauliString{n, {std::uint64_t{1} << (n - 1)}, {1}},
                                  std::complex<double>{0.25, 0.0});
    return observable;
}

bool same_bits(std::complex<double> a, std::complex<double> b) {
    return std::bit_cast<std::uint64_t>(a.real()) == std::bit_cast<std::uint64_t>(b.real()) &&
           std::bit_cast<std::uint64_t>(a.imag()) == std::bit_cast<std::uint64_t>(b.imag());
}

std::size_t scratch_files_in(const std::filesystem::path& directory) {
#if defined(_WIN32)
    const auto pid = static_cast<unsigned long>(_getpid());
#else
    const auto pid = static_cast<unsigned long>(getpid());
#endif
    // This process's files only: other test processes share the directory.
    const std::string prefix = "menos-scratch-" + std::to_string(pid) + "-";
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().filename().string().starts_with(prefix)) {
            ++count;
        }
    }
    return count;
}

// n = 14 with a 256 KiB budget: the 256 KiB state and its outputs do not fit,
// a window and the outputs do.
constexpr std::uint32_t kN = 14;
constexpr std::uint64_t kSmallBudget = std::uint64_t{256} << 10;

TEST(TierRequestTest, AJobWhoseStateDoesNotFitRunsOnTheDiskTierWithTheSameOutputs) {
    const Circuit circuit = entangled(kN, 3);
    const Observable observable = three_terms(kN);
    const std::filesystem::path directory = std::filesystem::current_path();
    const std::size_t before = scratch_files_in(directory);
    RunOptions in_memory{.shots = 500, .seed = 11};
    in_memory.expectation = observable;
    in_memory.backend = Backend::Cpu;
    const Result expected = run(circuit, in_memory);
    ASSERT_EQ(expected.plan().regions[0].tier, Tier::InMemory);

    RunOptions over = in_memory;
    over.memory_budget_bytes = kSmallBudget;
    over.scratch_directory = directory;
    const Result actual = run(circuit, over);
    ASSERT_EQ(actual.plan().strategy, Strategy::DirectDense);
    EXPECT_EQ(actual.plan().regions[0].tier, Tier::Disk);
    EXPECT_EQ(actual.counts(), expected.counts());
    ASSERT_TRUE(actual.expectation_value().has_value());
    EXPECT_TRUE(same_bits(*actual.expectation_value(), *expected.expectation_value()))
        << *actual.expectation_value() << " vs " << *expected.expectation_value();
    EXPECT_EQ(actual.peak_disk_bytes(), std::uint64_t{16} << kN);
    EXPECT_LE(actual.peak_host_bytes(), kSmallBudget);
    EXPECT_LE(actual.peak_host_bytes(), actual.plan().estimated_state_bytes);
    EXPECT_EQ(scratch_files_in(directory), before);

    const std::string report = actual.explain();
    EXPECT_NE(report.find(" tier=disk passes="), std::string::npos) << report;
    EXPECT_NE(report.find("engine=dense backend=cpu tier=disk"), std::string::npos) << report;
    EXPECT_NE(report.find(" scratch_directory=\"" + directory.string() + "\""), std::string::npos)
        << report;
    EXPECT_NE(report.find(" disk_available_bytes="), std::string::npos) << report;
    EXPECT_NE(report.find(" disk_write_bandwidth="), std::string::npos) << report;
    EXPECT_NE(report.find(" peak_disk_bytes=" + std::to_string(std::uint64_t{16} << kN)),
              std::string::npos)
        << report;
    // The same job previewed plans the same way.
    const Plan preview_plan = preview(circuit, over);
    EXPECT_EQ(preview_plan.regions[0].tier, Tier::Disk);
    EXPECT_EQ(preview_plan.regions[0].tier_cost.chunk_qubits,
              actual.plan().regions[0].tier_cost.chunk_qubits);
}

TEST(TierRequestTest, AnUnsetTierLeavesAFittingJobsReportByteIdentical) {
    const RunOptions options{.shots = 100, .seed = 1};
    ASSERT_FALSE(options.tier.has_value());
    ASSERT_FALSE(options.scratch_directory.has_value());
    const std::string report = preview(bell(), options).explain();
    EXPECT_EQ(report.find("tier"), std::string::npos) << report;
    EXPECT_EQ(report.find("disk"), std::string::npos) << report;
    EXPECT_EQ(run(bell(), options).explain().find("disk"), std::string::npos);
}

TEST(TierRequestTest, AForcedDiskTierRunsAFittingJobOnItAndNamesTheRequest) {
    // A circuit the planner runs on the in-memory dense engine when left
    // alone, so the two runs draw the same counts.
    const Circuit circuit = entangled(10, 3);
    RunOptions options{.shots = 100, .seed = 1};
    options.tier = Tier::Disk;
    options.scratch_directory = std::filesystem::current_path();
    const Result forced = run(circuit, options);
    const Result unforced =
        run(circuit, RunOptions{.shots = 100, .seed = 1, .backend = Backend::Cpu});
    ASSERT_EQ(unforced.plan().strategy, Strategy::DirectDense);
    ASSERT_EQ(unforced.plan().regions[0].tier, Tier::InMemory);
    EXPECT_EQ(forced.counts(), unforced.counts());
    ASSERT_EQ(forced.plan().regions.size(), 1U);
    EXPECT_EQ(forced.plan().regions[0].tier, Tier::Disk);
    EXPECT_EQ(forced.plan().regions[0].engine, Engine::Dense);
    EXPECT_EQ(forced.peak_disk_bytes(), std::uint64_t{16} << 10);
    const std::string report = forced.explain();
    EXPECT_NE(report.find("tier       disk (requested)\n"), std::string::npos) << report;
    EXPECT_NE(report.find("strategy=direct_dense feasible=true winner=true"), std::string::npos)
        << report;
    EXPECT_NE(report.find(" has no disk tier"), std::string::npos) << report;
    EXPECT_NE(report.find(" peak_disk_bytes=" + std::to_string(std::uint64_t{16} << 10)),
              std::string::npos)
        << report;
}

TEST(TierRequestTest, AForcedTierNeverFallsBack) {
    RunOptions disk{.shots = 100, .seed = 1};
    disk.tier = Tier::Disk;
    disk.scratch_directory = std::filesystem::current_path() / "no-such-directory-2312";
    EXPECT_THROW((void)run(bell(), disk), common::ResourceExceeded);

    RunOptions statevector{.shots = 0, .want_statevector = true, .seed = 1};
    statevector.tier = Tier::Disk;
    EXPECT_THROW((void)run(bell(), statevector), common::UnsupportedFeature);

    RunOptions offloaded{.shots = 100, .seed = 1};
    offloaded.tier = Tier::HostOffloaded;
    EXPECT_THROW((void)run(bell(), offloaded), common::UnsupportedFeature);

    RunOptions in_memory{.shots = 100, .seed = 1};
    in_memory.tier = Tier::InMemory;
    in_memory.memory_budget_bytes = kSmallBudget;
    try {
        (void)run(entangled(kN, 3), in_memory);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_EQ(std::string(e.what()).find("disk"), std::string::npos) << e.what();
    }
}

TEST(TierRequestTest, AStatevectorRequestOverTheBudgetIsRefusedWithoutTheTier) {
    RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
    options.memory_budget_bytes = kSmallBudget;
    try {
        (void)run(entangled(kN, 3), options);
        FAIL() << "expected ResourceExceeded";
    } catch (const common::ResourceExceeded& e) {
        EXPECT_NE(std::string(e.what()).find("direct dense simulation requires"), std::string::npos)
            << e.what();
    }
}

}  // namespace
}  // namespace menos
