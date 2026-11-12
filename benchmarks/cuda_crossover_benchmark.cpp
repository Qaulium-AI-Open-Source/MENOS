// The CPU-versus-CUDA crossover benchmark (docs/BUILD_PLAN.md Phase 10's
// "a benchmark shows the crossover qubit count", docs/adr/0014).
//
// The headline number is END-TO-END wall time including every transfer, not
// kernel time. Kernel time is reported alongside it, and the gap between the
// two columns is exactly the point: a kernel-only comparison would claim a win
// the user never receives. The Phase 10 gate says that if the GPU backend is
// not faster anywhere useful it is not to be shipped, and this is the
// measurement that decides it.
//
// Both output modes are swept, because they cost very different things:
//   statevector -- pays a 16 * 2^n device-to-host transfer;
//   expectation -- returns one complex per term and pays essentially none,
//                  which is where the device's advantage is largest.
//
// This is a benchmark, not a test (CLAUDE.md: benchmarks are not tests). It
// asserts nothing; it prints a table with the hardware and toolkit recorded.
//
// Setting MENOS_CROSSOVER_JSON=<path> also writes the same rows as JSON, for
// tools/dashboard/menos_dashboard.py. It is an environment variable, not a
// flag, because the shared main() hands argv to Google Benchmark, which rejects
// flags it does not know. A row that was not measured is null in the JSON,
// exactly as it is "(skipped)" or "(no VRAM)" in the table.
//
// Setting MENOS_CROSSOVER_REPEATS=<r> (Phase 16 step 1, docs/adr/0020 section
// 2) times every point r times after the warm-up. The CPU(s) and CUDA(s)
// columns, and cpu_seconds/cuda_seconds in the JSON, are then medians; the
// range and interquartile range are printed beside them and added to the JSON
// as *_min, *_max, *_q1 and *_q3. Unset (r = 1), the table and the JSON are
// exactly what they were before, so tools/dashboard/ reads either.

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "benchmark_spread.h"
#include "crossover_circuit.h"
#include "engines/dense/dense_engine.h"
#include "engines/dense_cuda/cuda_dense_engine.h"
#include "kernels/cuda/device_info.h"
#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/simulation_engine.h"
#include "plan/hardware_profile.h"

namespace {

using Clock = std::chrono::steady_clock;

using menos::benchmarks::kCrossoverLayers;
using menos::benchmarks::make_crossover_circuit;

menos::Observable single_z() {
    menos::PauliString pauli;
    pauli.num_qubits = 1;
    pauli.x_words = {0};
    pauli.z_words = {1};
    menos::Observable observable;
    observable.terms.emplace_back(pauli, std::complex<double>{1.0, 0.0});
    return observable;
}

// MENOS_CROSSOVER_REPEATS, or 1 when unset. 0 when set to anything other than
// an integer from 1 to 1000, which report() refuses rather than guessing.
int repeats_from_environment() {
    const char* const text = std::getenv("MENOS_CROSSOVER_REPEATS");
    if (text == nullptr || *text == '\0') {
        return 1;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*end != '\0' || value < 1 || value > 1000) {
        return 0;
    }
    return static_cast<int>(value);
}

// One warm-up run, then `repeats` timed runs.
menos::benchmarks::Spread time_run(const menos::SimulationEngine& engine,
                                   const menos::EngineExecutionRequest& request,
                                   int repeats) {
    static_cast<void>(engine.execute(request));  // warm
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const Clock::time_point start = Clock::now();
        static_cast<void>(engine.execute(request));
        samples.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return menos::benchmarks::spread(std::move(samples));
}

std::string seconds_text(double seconds) {
    return std::to_string(seconds).substr(0, 10);
}

// "min-max [q1-q3]", printed beside the medians when repeats > 1.
std::string range_text(const menos::benchmarks::Spread& spread) {
    char buffer[96];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "%.6f-%.6f [%.6f-%.6f]",
                  spread.min,
                  spread.max,
                  spread.q1,
                  spread.q3);
    return buffer;
}

void add_spread(nlohmann::json& row,
                const std::string& prefix,
                const menos::benchmarks::Spread& spread) {
    row[prefix + "_min"] = spread.min;
    row[prefix + "_max"] = spread.max;
    row[prefix + "_q1"] = spread.q1;
    row[prefix + "_q3"] = spread.q3;
}

// Prints one mode's table and returns the same rows as JSON.
nlohmann::json sweep(bool want_statevector, int repeats) {
    nlohmann::json rows = nlohmann::json::array();
    const menos::engines::dense::DenseEngine cpu;
    const menos::engines::dense_cuda::CudaDenseEngine gpu;
    const menos::Observable observable = single_z();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();

    std::printf("\n== output mode: %s ==\n", want_statevector ? "statevector" : "expectation");
    std::printf("%3s %12s %12s %12s %12s %12s %10s %9s",
                "n",
                "host_bytes",
                "dev_bytes",
                "work_units",
                "CPU(s)",
                "CUDA(s)",
                "speedup",
                "VRAM_ok");
    if (repeats > 1) {
        std::printf("  %-42s %-42s", "CPU min-max [q1-q3]", "CUDA min-max [q1-q3]");
    }
    std::printf("\n");

    for (std::uint32_t n = 4; n <= 28; n += 2) {
        const menos::Circuit circuit = make_crossover_circuit(n);
        const menos::EngineEstimateRequest estimate_request{
            n,
            &circuit.operations(),
            want_statevector,
            /*want_counts=*/false,
            !want_statevector,
            want_statevector ? nullptr : &observable,
        };
        const menos::CostEstimate cpu_estimate = cpu.estimate(estimate_request);
        const menos::CostEstimate gpu_estimate = gpu.estimate(estimate_request);

        const bool vram_ok =
            gpu_estimate.feasible && gpu_estimate.device.device_bytes <= device_budget;
        const std::uint64_t host_bytes = cpu_estimate.estimated_peak_bytes;
        const std::uint64_t dev_bytes =
            gpu_estimate.feasible ? gpu_estimate.device.device_bytes : (std::uint64_t{16} << n);

        // The host budget is generous on purpose; VRAM is the binding limit
        // here and the point is to show where it binds.
        const menos::EngineExecutionRequest request{
            .num_qubits = n,
            .operations = circuit.operations(),
            .want_statevector = want_statevector,
            .want_counts = false,
            .want_expectation = !want_statevector,
            .shots = 0,
            .measure_targets = {},
            .bitstring_width = 0,
            .seed = 0,
            .global_phase = 0.0,
            .budget_bytes = std::uint64_t{1} << 36,
            .is_sole_region = true,
            .observable = want_statevector ? nullptr : &observable,
        };

        // Above roughly 26 qubits the CPU run takes minutes, which makes the
        // table impractical to regenerate; the GPU is reported alone there and
        // the CPU column is left blank rather than guessed.
        const bool run_cpu = n <= 24;
        const std::optional<menos::benchmarks::Spread> cpu_spread =
            run_cpu ? std::optional(time_run(cpu, request, repeats)) : std::nullopt;
        const std::optional<menos::benchmarks::Spread> gpu_spread =
            vram_ok ? std::optional(time_run(gpu, request, repeats)) : std::nullopt;
        const double cpu_seconds = cpu_spread ? cpu_spread->median : -1.0;
        const double gpu_seconds = gpu_spread ? gpu_spread->median : -1.0;

        std::string speedup = "-";
        if (cpu_seconds > 0.0 && gpu_seconds > 0.0) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.2fx", cpu_seconds / gpu_seconds);
            speedup = buffer;
        }
        std::printf("%3u %12llu %12llu %12llu %12s %12s %10s %9s",
                    n,
                    static_cast<unsigned long long>(host_bytes),
                    static_cast<unsigned long long>(dev_bytes),
                    static_cast<unsigned long long>(cpu_estimate.estimated_work_units),
                    run_cpu ? seconds_text(cpu_seconds).c_str() : "(skipped)",
                    vram_ok ? seconds_text(gpu_seconds).c_str() : "(no VRAM)",
                    speedup.c_str(),
                    vram_ok ? "yes" : "NO");
        if (repeats > 1) {
            std::printf("  %-42s %-42s",
                        cpu_spread ? range_text(*cpu_spread).c_str() : "-",
                        gpu_spread ? range_text(*gpu_spread).c_str() : "-");
        }
        std::printf("\n");
        std::fflush(stdout);

        nlohmann::json row = {
            {"n", n},
            {"host_bytes", host_bytes},
            {"dev_bytes", dev_bytes},
            {"work_units", cpu_estimate.estimated_work_units},
            {"cpu_seconds", run_cpu ? nlohmann::json(cpu_seconds) : nlohmann::json(nullptr)},
            {"cuda_seconds", vram_ok ? nlohmann::json(gpu_seconds) : nlohmann::json(nullptr)},
            {"vram_ok", vram_ok},
        };
        if (repeats > 1) {
            if (cpu_spread) {
                add_spread(row, "cpu_seconds", *cpu_spread);
            }
            if (gpu_spread) {
                add_spread(row, "cuda_seconds", *gpu_spread);
            }
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// Writes the report to $MENOS_CROSSOVER_JSON when it is set; otherwise nothing.
void write_json(const nlohmann::json& document) {
    const char* const path = std::getenv("MENOS_CROSSOVER_JSON");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream out(path);
    out << document.dump(2) << "\n";
    out.close();
    std::printf("JSON written     : %s%s\n", path, out ? "" : " (WRITE FAILED)");
}

void report() {
    const menos::kernels::cuda::DeviceInfo& info = menos::kernels::cuda::device_info();
    std::printf("\n=== MENOS CPU vs CUDA crossover ===\n");
    const int repeats = repeats_from_environment();
    if (repeats == 0) {
        std::printf(
            "MENOS_CROSSOVER_REPEATS must be an integer from 1 to 1000 -- nothing measured\n");
        return;
    }
    nlohmann::json document = {{"schema_version", 1}, {"device_present", info.present}};
    if (!info.present) {
        std::printf("no CUDA device available -- nothing measured\n");
        write_json(document);
        return;
    }
    document["gpu"] = {
        {"name", info.name},
        {"compute_capability",
         std::to_string(info.compute_capability_major) + "." +
             std::to_string(info.compute_capability_minor)},
        {"multiprocessors", info.multiprocessor_count},
        {"total_memory_bytes", info.total_memory_bytes},
        {"free_memory_bytes", info.free_memory_bytes},
    };
    std::printf("GPU              : %s (compute capability %d.%d, %d SMs)\n",
                info.name.c_str(),
                info.compute_capability_major,
                info.compute_capability_minor,
                info.multiprocessor_count);
    std::printf("device memory    : %.0f MiB total, %.0f MiB free\n",
                static_cast<double>(info.total_memory_bytes) / 1048576.0,
                static_cast<double>(info.free_memory_bytes) / 1048576.0);
#ifdef __CUDACC_VER_MAJOR__
    std::printf("CUDA toolkit     : %d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    document["cuda_toolkit"] =
        std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__);
#endif

    const menos::plan::HardwareProfile& hw = menos::plan::calibrated_hardware_profile();
    std::printf("calibration      : host %.4g work_units/s, device %.4g work_units/s\n",
                hw.work_units_per_second,
                hw.cuda_device_work_units_per_second);
    std::printf("                   H2D %.4g B/s, D2H %.4g B/s, launch %.4g s, execution %.4g s\n",
                hw.cuda_host_to_device_bytes_per_second,
                hw.cuda_device_to_host_bytes_per_second,
                hw.cuda_kernel_launch_seconds,
                hw.cuda_execution_seconds);
    std::printf(
        "circuit          : %d layers of (H on all qubits, T on q0, CX on alternating "
        "pairs)\n",
        kCrossoverLayers);
    std::printf("timing           : end-to-end SimulationEngine::execute(), transfers included\n");
    if (repeats > 1) {
        std::printf(
            "repeats          : %d per point after one warm-up; CPU(s) and CUDA(s) are medians\n",
            repeats);
        document["repeats"] = repeats;
    }
    document["calibration"] = {
        {"host_work_units_per_second", hw.work_units_per_second},
        {"device_work_units_per_second", hw.cuda_device_work_units_per_second},
        {"h2d_bytes_per_second", hw.cuda_host_to_device_bytes_per_second},
        {"d2h_bytes_per_second", hw.cuda_device_to_host_bytes_per_second},
        {"kernel_launch_seconds", hw.cuda_kernel_launch_seconds},
        {"execution_seconds", hw.cuda_execution_seconds},
    };
    document["circuit"] = {
        {"layers", kCrossoverLayers},
        {"description", "H on all qubits, T on q0, CX on alternating pairs, per layer"},
    };
    document["timing"] = "end-to-end SimulationEngine::execute(), transfers included";

    document["statevector"] = sweep(/*want_statevector=*/true, repeats);
    document["expectation"] = sweep(/*want_statevector=*/false, repeats);
    std::printf("\n");
    write_json(document);
}

}  // namespace

// Registered as a Google Benchmark fixture so it runs under the same harness as
// the project's other benchmarks, but the work is a single reported sweep
// rather than a repeated micro-measurement.
static void BM_CudaCrossoverReport(benchmark::State& state) {
    for (auto _ : state) {
        report();
        state.SkipWithError("report-only benchmark: see the printed table");
        break;
    }
}
BENCHMARK(BM_CudaCrossoverReport)->Iterations(1)->Unit(benchmark::kSecond);
