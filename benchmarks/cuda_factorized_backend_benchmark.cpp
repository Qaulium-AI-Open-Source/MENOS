// Phase 17 step 4a (docs/adr/0021 section 5): does choosing CPU or CUDA per
// region pay?
//
// Benchmark-only. It measures, and decides nothing:
//
//   (a) Factorized plans. Circuits of 2 or 3 independent blocks, each block a
//       scrambling circuit on its own qubits (below), every qubit measured.
//       menos::preview plans each circuit: a Factorized plan with one DENSE
//       region per block, which is the only case where a region's backend is
//       a choice at all (an MPS or stabilizer region has no CUDA engine). The plan is then executed
//       through runtime::Runtime::execute under EVERY assignment of CPU/CUDA to its regions, so the
//       engines run each region exactly as in production. Reported per assignment: the median
//       end-to-end time (10 repeats, spread beside it), and the best assignment against all-CPU and
//       the planner's own (unforced) plan.
//   (b) Cut fragments. A CutAndRecombine plan evaluates each fragment many
//       times, and the cut executor chooses its engines per evaluation, not
//       from region.backend. So one fragment-sized evaluation (the dense
//       expectation value of one scrambling block) is timed on each backend
//       with RunOptions::backend, for fragment sizes 8-18, and the plan is
//       checked to be dense on both.
//
// Why not the crossover family: its nearest-neighbour CX layers are cheap for
// the MPS engine, so the planner gives such blocks to MPS, not dense (found in
// the first 4a run, docs/adr/0021 section 10).
//
// Setting MENOS_FACTORIZED_BACKEND_JSON=<path> also writes the tables as JSON.

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
// std::getenv is the portable call for the environment variable read below;
// MSVC warns about it as it does for every C standard call with a _s variant
// (benchmarks/tiered_memory_benchmark.cpp takes the same route).
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "benchmark_spread.h"
#include "kernels/cuda/device_info.h"
#include "menos/menos.h"
#include "menos_benchmark_environment.h"
#include "plan/job_spec.h"
#include "runtime/runtime.h"

namespace {

using Clock = std::chrono::steady_clock;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;

constexpr int kRepetitions = 10;
constexpr std::uint64_t kShots = 1000;
constexpr std::uint64_t kSeed = 1;

nlohmann::json spread_json(const Spread& s) {
    return {{"median", s.median}, {"min", s.min}, {"max", s.max}, {"q1", s.q1}, {"q3", s.q3}};
}

// One scrambling block of n qubits at `offset`: 6 layers of H and T on every
// qubit, then CX on a random perfect pairing that changes every layer (fixed
// seed). The pairings are long-range and connect the block, so its MPS bond
// dimension saturates and the planner simulates it with the dense engine.
// offset and n are a qubit index and a qubit count, named at both call sites.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void append_scrambling_block(menos::Circuit& circuit, std::uint32_t offset, std::uint32_t n) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    constexpr int kLayers = 6;
    std::mt19937 rng(1700 + n);
    std::vector<std::uint32_t> order(n);
    for (int layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(menos::Opcode::H, {static_cast<menos::QubitId>(offset + q)});
            circuit.append(menos::Opcode::T, {static_cast<menos::QubitId>(offset + q)});
        }
        for (std::uint32_t q = 0; q < n; ++q) {
            order[q] = q;
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (std::uint32_t q = 0; q + 1 < n; q += 2) {
            circuit.append(menos::Opcode::CX,
                           {static_cast<menos::QubitId>(offset + order[q]),
                            static_cast<menos::QubitId>(offset + order[q + 1])});
        }
    }
}

// Independent scrambling blocks, then every qubit measured.
menos::Circuit make_blocks(const std::vector<std::uint32_t>& sizes) {
    std::uint32_t total = 0;
    for (const std::uint32_t size : sizes) {
        total += size;
    }
    menos::Circuit circuit(total, total);
    std::uint32_t offset = 0;
    for (const std::uint32_t size : sizes) {
        append_scrambling_block(circuit, offset, size);
        offset += size;
    }
    for (menos::QubitId q = 0; q < total; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

std::string assignment_name(const menos::Plan& plan) {
    std::string name;
    for (const menos::Region& region : plan.regions) {
        name += region.backend == menos::Backend::Cuda ? "G" : "C";
    }
    return name;
}

Spread time_execution(const menos::Plan& plan, const menos::plan::JobSpec& job_spec) {
    const menos::runtime::Runtime runtime;
    static_cast<void>(runtime.execute(plan, job_spec));  // warm-up
    std::vector<double> samples;
    for (int r = 0; r < kRepetitions; ++r) {
        const Clock::time_point start = Clock::now();
        menos::Result result = runtime.execute(plan, job_spec);
        samples.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        benchmark::DoNotOptimize(result);
    }
    return spread(std::move(samples));
}

nlohmann::json factorized_section() {
    std::printf("\n== (a) Factorized plans: every CPU/CUDA assignment of the regions ==\n");
    std::printf("assignment letters follow region order: C = CPU, G = CUDA\n");
    const std::vector<std::vector<std::uint32_t>> configurations = {
        {12, 22}, {16, 20}, {20, 22}, {10, 16, 22}, {14, 18, 22}};
    nlohmann::json rows = nlohmann::json::array();
    for (const std::vector<std::uint32_t>& sizes : configurations) {
        const menos::Circuit circuit = make_blocks(sizes);
        const menos::RunOptions options{.shots = kShots, .seed = kSeed};
        const menos::Plan planned = menos::preview(circuit, options);
        std::string label;
        for (const std::uint32_t size : sizes) {
            label += (label.empty() ? "" : "+") + std::to_string(size);
        }
        if (planned.strategy != menos::Strategy::Factorized) {
            std::printf("%s: planner chose a non-factorized plan; skipped\n", label.c_str());
            rows.push_back({{"blocks", sizes}, {"factorized", false}});
            continue;
        }
        const menos::plan::JobSpec job_spec{.circuit = circuit,
                                            .shots = kShots,
                                            .want_statevector = false,
                                            .seed = kSeed,
                                            .memory_budget_bytes = std::nullopt,
                                            .expectation = std::nullopt};
        std::printf("\n%s qubits (%zu regions; planner's own assignment: %s)\n",
                    label.c_str(),
                    planned.regions.size(),
                    assignment_name(planned).c_str());
        std::printf("  %-10s %12s %24s\n", "assignment", "median (s)", "min-max (s)");
        nlohmann::json assignments = nlohmann::json::array();
        const std::size_t count = planned.regions.size();
        double all_cpu = 0.0;
        double planner = 0.0;
        double best = 0.0;
        std::string best_name;
        for (std::size_t mask = 0; mask < (std::size_t{1} << count); ++mask) {
            menos::Plan forced = planned;
            for (std::size_t i = 0; i < count; ++i) {
                forced.regions[i].backend =
                    ((mask >> i) & 1U) != 0 ? menos::Backend::Cuda : menos::Backend::Cpu;
            }
            const Spread timing = time_execution(forced, job_spec);
            const std::string name = assignment_name(forced);
            std::printf("  %-10s %12.6f %11.6f-%-11.6f\n",
                        name.c_str(),
                        timing.median,
                        timing.min,
                        timing.max);
            std::fflush(stdout);
            if (mask == 0) {
                all_cpu = timing.median;
            }
            if (best_name.empty() || timing.median < best) {
                best = timing.median;
                best_name = name;
            }
            if (name == assignment_name(planned)) {
                planner = timing.median;
            }
            assignments.push_back({{"assignment", name}, {"seconds", spread_json(timing)}});
        }
        std::vector<std::uint32_t> region_qubits;
        region_qubits.reserve(planned.regions.size());
        for (const menos::Region& region : planned.regions) {
            region_qubits.push_back(static_cast<std::uint32_t>(region.local_to_global.size()));
        }
        std::printf("  best %s: %.2fx faster than all-CPU\n", best_name.c_str(), all_cpu / best);
        // The Phase 17 gate figure (docs/BUILD_PLAN.md criterion 4): the
        // planner's own assignment against all-CPU, from this same run.
        std::printf("  planner %s: %.2fx faster than all-CPU\n",
                    assignment_name(planned).c_str(),
                    all_cpu / planner);
        rows.push_back({{"blocks", sizes},
                        {"factorized", true},
                        {"region_qubits", region_qubits},
                        {"planner_assignment", assignment_name(planned)},
                        {"assignments", assignments},
                        {"best_assignment", best_name},
                        {"best_over_all_cpu", all_cpu / best},
                        {"planner_over_all_cpu", all_cpu / planner}});
    }
    return rows;
}

nlohmann::json fragment_section() {
    std::printf("\n== (b) One fragment-sized evaluation (dense expectation) on each backend ==\n");
    std::printf("%3s %12s %12s %10s\n", "n", "CPU (s)", "CUDA (s)", "CPU/CUDA");
    nlohmann::json rows = nlohmann::json::array();
    for (std::uint32_t n = 8; n <= 18; n += 2) {
        menos::Circuit circuit(n, 0);
        append_scrambling_block(circuit, 0, n);
        menos::PauliString z_all{.num_qubits = n,
                                 .x_words = std::vector<std::uint64_t>((n + 63) / 64, 0),
                                 .z_words = std::vector<std::uint64_t>((n + 63) / 64, 0)};
        for (std::uint32_t q = 0; q < n; ++q) {
            z_all.z_words[q / 64] |= std::uint64_t{1} << (q % 64);
        }
        menos::Observable observable;
        observable.terms.emplace_back(z_all, std::complex<double>{1.0, 0.0});
        Spread timings[2];
        for (const int b : {0, 1}) {
            menos::RunOptions options{.shots = 0, .seed = kSeed, .expectation = observable};
            options.backend = b == 0 ? menos::Backend::Cpu : menos::Backend::Cuda;
            const menos::Result warm = menos::run(circuit, options);  // warm-up
            if (warm.plan().strategy != menos::Strategy::DirectDense) {
                std::printf("%3u: not a dense plan on %s; the comparison is invalid\n",
                            n,
                            b == 0 ? "CPU" : "CUDA");
            }
            std::vector<double> samples;
            for (int r = 0; r < kRepetitions; ++r) {
                const Clock::time_point start = Clock::now();
                menos::Result result = menos::run(circuit, options);
                samples.push_back(std::chrono::duration<double>(Clock::now() - start).count());
                benchmark::DoNotOptimize(result);
            }
            timings[b] = spread(std::move(samples));
        }
        std::printf("%3u %12.6f %12.6f %9.2fx\n",
                    n,
                    timings[0].median,
                    timings[1].median,
                    timings[0].median / timings[1].median);
        std::fflush(stdout);
        rows.push_back({{"n", n},
                        {"cpu_seconds", spread_json(timings[0])},
                        {"cuda_seconds", spread_json(timings[1])}});
    }
    return rows;
}

void report() {
    const menos::kernels::cuda::DeviceInfo& info = menos::kernels::cuda::device_info();
    std::printf("\n=== MENOS per-region backend benchmark (Phase 17 step 4a) ===\n");
    nlohmann::json document = {{"schema_version", 1}, {"device_present", info.present}};
    if (!info.present) {
        std::printf("no CUDA device available -- nothing measured\n");
    } else {
        std::printf("environment      : %s\n", menos_benchmark::describe_environment().c_str());
        std::printf("GPU              : %s\n", info.name.c_str());
        std::printf("repetitions      : %d per point after a warm-up; shots %llu\n",
                    kRepetitions,
                    static_cast<unsigned long long>(kShots));
        document["environment"] = menos_benchmark::describe_environment();
        document["gpu"] = info.name;
        document["repetitions"] = kRepetitions;
        document["factorized"] = factorized_section();
        document["fragments"] = fragment_section();
    }
    const char* const path = std::getenv("MENOS_FACTORIZED_BACKEND_JSON");
    if (path != nullptr && *path != '\0') {
        std::ofstream out(path);
        out << document.dump(2) << "\n";
        std::printf("\nJSON written     : %s%s\n", path, out ? "" : " (WRITE FAILED)");
    }
}

}  // namespace

// Report-only, like BM_CudaCrossoverReport.
static void BM_FactorizedBackendReport(benchmark::State& state) {
    for (auto _ : state) {
        report();
        state.SkipWithError("report-only benchmark: see the printed tables");
        break;
    }
}
BENCHMARK(BM_FactorizedBackendReport)->Iterations(1)->Unit(benchmark::kSecond);
