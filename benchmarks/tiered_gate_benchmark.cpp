// Phase 23 step 5 (docs/adr/0027 section 6): the disk tier's gate-scale runs,
// through the public API, for criteria 1-3 of section 9 Q9.
//
//   * overlap: at the largest size the in-memory dense engine still runs on
//     this machine (n = 29 under a 16 GB budget), the same circuit, shots,
//     seed and 20-term observable on the in-memory engine and on the disk
//     tier forced; the counts must be equal and the expectation value the
//     same bits (criterion 2).
//   * reach: at the smallest size the in-memory engine cannot run (n = 31
//     under a 16 GB budget; n = 30 under WSL's 12 GB), the planner left to
//     itself must choose the disk tier (criterion 1), and the wall time must
//     be within 2x of the plan's own predicted seconds (criterion 3, the
//     Q7 factor), measured on the same run.
//
// The circuit is the first MENOS_TIERED_GATE_GATES (100) operations of the
// crossover family (benchmarks/crossover_circuit.h) at each size, every qubit
// measured. Not a Google Benchmark binary: one timed run each, no repetitions,
// because a run writes a 16-32 GiB scratch file. The environment:
//   MENOS_TIERED_GATE_JSON            the report (JSON), required
//   MENOS_TIERED_SCRATCH_DIR          the scratch directory, required
//   MENOS_TIERED_GATE_REACH_QUBITS    n for the reach run (31)
//   MENOS_TIERED_GATE_OVERLAP_QUBITS  n for the overlap runs (29)
//   MENOS_TIERED_GATE_SHOTS           shots (1000)
//   MENOS_TIERED_GATE_GATES           operations kept (100)
//   MENOS_TIERED_GATE_SKIP            "overlap" or "reach" to skip one

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
// std::getenv is the portable call for the environment variables above; MSVC
// warns about it as it does for every C standard call with a _s variant
// (benchmarks/tiered_memory_benchmark.cpp takes the same route).
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <bit>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/error.h"
#include "menos/menos.h"
#include "menos_benchmark_environment.h"

namespace {

using Clock = std::chrono::steady_clock;

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
    return value == nullptr ? fallback : std::string(value);
}

// `gates` operations of the random-pairs family of step 1 (RX on a random
// qubit, CX on a random distinct pair, alternating), every qubit measured to
// its own clbit. Random pairs, not the crossover family: a hundred
// operations of a layered circuit are shallow enough for the MPS engine to
// take the job in memory at any size, which is a different plan, not the
// dense state the gate is about; random pairs entangle across every cut
// after a few dozen gates, so the dense algorithm is the only one that can
// run the circuit. Drawn from std::mt19937_64's standardised output, so the
// circuit is the same on every platform.
menos::Circuit gate_circuit(std::uint32_t n, std::size_t gates) {
    std::mt19937_64 rng(2301);
    menos::Circuit circuit(n, n);
    // A CX chain first, so the interaction graph is one component whatever
    // the random pairs touch: a disconnected graph is a Factorized plan of
    // smaller dense regions, not the one dense state the gate is about (the
    // first attempt at n = 29 found exactly that).
    for (menos::QubitId q = 0; q + 1 < n && q < gates; ++q) {
        circuit.append(menos::Opcode::CX, {q, static_cast<menos::QubitId>(q + 1)});
    }
    for (std::size_t g = circuit.operations().size(); g < gates; ++g) {
        const auto a = static_cast<menos::QubitId>(rng() % n);
        if (g % 2 == 0) {
            const double angle =
                static_cast<double>(rng() >> 11) * (6.283185307179586 / 9007199254740992.0);
            circuit.append(menos::Opcode::RX, {a}, {angle});
        } else {
            auto b = static_cast<menos::QubitId>(rng() % n);
            while (b == a) {
                b = static_cast<menos::QubitId>(rng() % n);
            }
            circuit.append(menos::Opcode::CX, {a, b});
        }
    }
    for (menos::QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

// Twenty terms: Z-only, X-only, a single high X bit, and mixed masks from a
// fixed multiplicative sequence, so several terms pair chunks across the
// global qubits.
menos::Observable twenty_terms(std::uint32_t n) {
    const std::uint64_t all = (std::uint64_t{1} << n) - 1;
    menos::Observable observable;
    std::uint64_t x = 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = 0xC2B2AE3D27D4EB4FULL;
    for (int k = 0; k < 20; ++k) {
        std::uint64_t x_mask = 0;
        std::uint64_t z_mask = 0;
        switch (k % 4) {
            case 0:
                z_mask = z & all;
                break;
            case 1:
                x_mask = x & all;
                break;
            case 2:
                x_mask = std::uint64_t{1} << (n - 1 - static_cast<std::uint32_t>(k % n));
                z_mask = z & all;
                break;
            default:
                x_mask = x & all;
                z_mask = z & all;
                break;
        }
        observable.terms.emplace_back(menos::PauliString{n, {x_mask}, {z_mask}},
                                      std::complex<double>{0.5 + 0.01 * k, 0.0});
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        z = z * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    return observable;
}

std::size_t scratch_files_in(const std::filesystem::path& directory) {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().filename().string().starts_with("menos-scratch-")) {
            ++count;
        }
    }
    return count;
}

// FNV-1a over the counts' keys and values, so two runs' histograms can be
// compared across processes.
std::string counts_digest(const std::map<menos::BitString, std::uint64_t>& counts) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&h](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (8 * i)) & 0xFF;
            h *= 1099511628211ULL;
        }
    };
    for (const auto& [key, value] : counts) {
        mix(key.num_bits());
        for (std::uint32_t bit = 0; bit < key.num_bits(); ++bit) {
            if (key.test(bit)) {
                mix(bit);
            }
        }
        mix(value);
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h));
    return text;
}

nlohmann::json describe(const menos::Result& result, double wall_seconds) {
    const menos::Plan& plan = result.plan();
    nlohmann::json j;
    const menos::CandidateSummary* winner = nullptr;
    for (const menos::CandidateSummary& c : plan.candidates) {
        if (c.is_winner) {
            winner = &c;
        }
    }
    j["strategy"] = plan.strategy == menos::Strategy::DirectDense ? "direct_dense" : "other";
    j["tier"] = plan.regions.empty()                        ? "none"
                : plan.regions[0].tier == menos::Tier::Disk ? "disk"
                                                            : "in_memory";
    if (winner != nullptr) {
        j["passes"] = winner->tier_cost.passes;
        j["chunk_qubits"] = winner->tier_cost.chunk_qubits;
        j["window_qubits"] = winner->tier_cost.window_qubits;
        j["link_bytes"] = winner->tier_cost.link_bytes;
        j["file_bytes"] = winner->tier_cost.file_bytes;
        j["predicted_seconds"] = winner->estimated_seconds;
        j["estimated_peak_bytes"] = winner->estimated_peak_bytes;
    }
    j["wall_seconds"] = wall_seconds;
    j["execute_seconds"] = result.execute_seconds();
    j["sample_seconds"] = result.sample_seconds();
    j["peak_host_bytes"] = result.peak_host_bytes();
    j["peak_disk_bytes"] = result.peak_disk_bytes();
    j["counts_distinct"] = result.counts().size();
    j["counts_digest"] = counts_digest(result.counts());
    if (result.expectation_value().has_value()) {
        const std::complex<double> value = result.expectation_value().value_or({});
        j["expectation_real"] = value.real();
        j["expectation_imag"] = value.imag();
        j["expectation_real_bits"] = std::bit_cast<std::uint64_t>(value.real());
        j["expectation_imag_bits"] = std::bit_cast<std::uint64_t>(value.imag());
    }
    if (plan.disk_write_bytes_per_second.has_value()) {
        j["disk_write_bytes_per_second"] = *plan.disk_write_bytes_per_second;
        j["disk_read_bytes_per_second"] = *plan.disk_read_bytes_per_second;
    }
    if (plan.disk_available_bytes.has_value()) {
        j["disk_available_bytes"] = *plan.disk_available_bytes;
    }
    j["explain"] = result.explain();
    return j;
}

menos::Result timed_run(const menos::Circuit& circuit,
                        const menos::RunOptions& options,
                        double& wall_seconds) {
    const auto start = Clock::now();
    menos::Result result = menos::run(circuit, options);
    wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

}  // namespace

int main() {
    const std::string json_path = env_or("MENOS_TIERED_GATE_JSON", "");
    const std::filesystem::path scratch(env_or("MENOS_TIERED_SCRATCH_DIR", ""));
    if (json_path.empty() || scratch.empty()) {
        std::cerr << "MENOS_TIERED_GATE_JSON and MENOS_TIERED_SCRATCH_DIR are required\n";
        return 2;
    }
    const auto reach_qubits =
        static_cast<std::uint32_t>(std::stoul(env_or("MENOS_TIERED_GATE_REACH_QUBITS", "31")));
    const auto overlap_qubits =
        static_cast<std::uint32_t>(std::stoul(env_or("MENOS_TIERED_GATE_OVERLAP_QUBITS", "29")));
    const std::uint64_t shots = std::stoull(env_or("MENOS_TIERED_GATE_SHOTS", "1000"));
    const std::size_t gates = std::stoul(env_or("MENOS_TIERED_GATE_GATES", "100"));
    const std::string skip = env_or("MENOS_TIERED_GATE_SKIP", "");
    // A memory budget for the reach run, so the driver can be smoke-tested at
    // a small size; unset for the gate, where the default budget is the
    // claim.
    const std::string budget_text = env_or("MENOS_TIERED_GATE_BUDGET_BYTES", "");
    const std::optional<std::uint64_t> budget =
        budget_text.empty() ? std::nullopt : std::optional<std::uint64_t>(std::stoull(budget_text));

    nlohmann::json report;
    report["environment"] = menos_benchmark::describe_environment();
    report["scratch_directory"] = scratch.string();
    report["shots"] = shots;
    report["gates"] = gates;
    std::filesystem::create_directories(scratch);
    const std::size_t files_before = scratch_files_in(scratch);
    int failures = 0;

    const auto options_for = [&](std::optional<menos::Tier> tier) {
        menos::RunOptions options{.shots = shots, .seed = 2311};
        options.backend = menos::Backend::Cpu;
        options.tier = tier;
        options.scratch_directory = scratch;
        return options;
    };

    if (skip != "overlap") {
        std::cerr << "== overlap at n = " << overlap_qubits << "\n";
        const menos::Circuit circuit = gate_circuit(overlap_qubits, gates);
        const menos::Observable observable = twenty_terms(overlap_qubits);
        report["overlap"]["qubits"] = overlap_qubits;
        report["overlap"]["operations"] = circuit.operations().size();

        menos::RunOptions in_memory = options_for(menos::Tier::InMemory);
        in_memory.expectation = observable;
        double wall = 0.0;
        std::cerr << "   in memory...\n";
        const menos::Result memory_result = timed_run(circuit, in_memory, wall);
        report["overlap"]["in_memory"] = describe(memory_result, wall);
        std::cerr << "   " << wall << " s\n";

        menos::RunOptions disk = options_for(menos::Tier::Disk);
        disk.expectation = observable;
        std::cerr << "   disk tier (forced)...\n";
        const menos::Result disk_result = timed_run(circuit, disk, wall);
        report["overlap"]["disk"] = describe(disk_result, wall);
        std::cerr << "   " << wall << " s\n";

        const bool same_counts = memory_result.counts() == disk_result.counts();
        const std::complex<double> a = memory_result.expectation_value().value_or({});
        const std::complex<double> b = disk_result.expectation_value().value_or({});
        const bool same_expectation =
            std::bit_cast<std::uint64_t>(a.real()) == std::bit_cast<std::uint64_t>(b.real()) &&
            std::bit_cast<std::uint64_t>(a.imag()) == std::bit_cast<std::uint64_t>(b.imag());
        report["overlap"]["bitwise_counts"] = same_counts;
        report["overlap"]["bitwise_expectation"] = same_expectation;
        // The reference must be the in-memory dense engine, not another
        // algorithm the planner found cheaper.
        const bool dense_in_memory =
            memory_result.plan().strategy == menos::Strategy::DirectDense &&
            !memory_result.plan().regions.empty() &&
            memory_result.plan().regions[0].tier == menos::Tier::InMemory;
        report["overlap"]["in_memory_is_dense"] = dense_in_memory;
        report["overlap"]["disk_tier_used"] =
            !disk_result.plan().regions.empty() &&
            disk_result.plan().regions[0].tier == menos::Tier::Disk;
        report["overlap"]["scratch_files_after"] = scratch_files_in(scratch) - files_before;
        if (!same_counts || !same_expectation || !dense_in_memory) {
            std::cerr << "   MISMATCH: counts " << same_counts << " expectation "
                      << same_expectation << " dense in memory " << dense_in_memory << "\n";
            ++failures;
        }
    }

    if (skip != "reach") {
        std::cerr << "== reach at n = " << reach_qubits << "\n";
        const menos::Circuit circuit = gate_circuit(reach_qubits, gates);
        menos::RunOptions options = options_for(std::nullopt);
        options.expectation = twenty_terms(reach_qubits);
        options.memory_budget_bytes = budget;
        report["reach"]["memory_budget_bytes"] =
            budget.has_value() ? nlohmann::json(*budget) : nlohmann::json("default");
        report["reach"]["qubits"] = reach_qubits;
        report["reach"]["operations"] = circuit.operations().size();
        try {
            double wall = 0.0;
            const menos::Result result = timed_run(circuit, options, wall);
            nlohmann::json run = describe(result, wall);
            const bool reached = !result.plan().regions.empty() &&
                                 result.plan().regions[0].tier == menos::Tier::Disk;
            run["disk_tier_chosen"] = reached;
            const double predicted = run.value("predicted_seconds", -1.0);
            run["measured_over_predicted"] = predicted > 0.0 ? wall / predicted : -1.0;
            run["within_factor_two"] =
                predicted > 0.0 && wall <= 2.0 * predicted && predicted <= 2.0 * wall;
            run["scratch_files_after"] = scratch_files_in(scratch) - files_before;
            report["reach"]["run"] = run;
            std::cerr << "   " << wall << " s, predicted " << predicted << " s, tier "
                      << (reached ? "disk" : "NOT disk") << "\n";
            if (!reached || !run["within_factor_two"].get<bool>()) {
                ++failures;
            }
        } catch (const menos::common::Error& e) {
            report["reach"]["error"] = e.what();
            std::cerr << "   refused: " << e.what() << "\n";
            ++failures;
        }
    }

    std::ofstream out(json_path);
    out << report.dump(2) << "\n";
    std::cerr << "report: " << json_path << "\n";
    return failures == 0 ? 0 : 1;
}
