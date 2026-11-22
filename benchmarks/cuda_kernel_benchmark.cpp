// The Phase 16 CUDA kernel benchmark (docs/adr/0020 sections 2 and 11).
//
// It measures three things, on the crossover circuit family
// (crossover_circuit.h):
//
//   (a) the shipped kernels::cuda::apply_matrix (specialised kernels for
//       k = 1 and 2) and the probe (hand-written H and CX kernels,
//       cuda_probe_kernels.cu, in the style of the probe docs/adr/0014
//       section 16.3 recorded), each against the generic kernel
//       kernels::cuda::apply_matrix_generic, in the same run. Every path is
//       driven as CudaDenseEngine drives apply_matrix, launching and
//       synchronizing once per gate. The figures of merit are generic /
//       specialised (the Phase 16 gate, docs/BUILD_PLAN.md) and generic /
//       probe (owner, docs/adr/0020 Q2), "gates only" and "end to end"
//       (allocate, initialize, gates, copy back, free). Whether the
//       specialised path's final state is bitwise identical to the generic
//       path's is printed with each row;
//   (b) one gate's cost in parts: the host wall time and CUDA-event span of
//       one generic and one specialised call, the probe's, an empty launch
//       plus synchronize, and the two constant-memory copies the generic
//       kernel makes, for k = 1 (H) and k = 2 (CX);
//   (c) device memory bandwidth: a device-to-device copy and a kernel that
//       reads and writes every amplitude once, which bound what a gate sweep
//       can cost.
//
// Every point is timed kRepetitions times after a warm-up, and the spread is
// reported beside the median (benchmark_spread.h). This is a benchmark, not a
// test: it asserts nothing. The largest difference between the probe's and
// the generic path's final states is printed, because the probe computes H
// as (a +- b) / sqrt(2) and is therefore not bitwise identical to the
// generic kernel; a large difference would mean the probe is wrong, not
// fast.
//
// Setting MENOS_CUDA_KERNEL_JSON=<path> also writes every table as JSON, for
// docs/benchmarks/phase-16-baseline/ and phase-16-gate/. As in
// cuda_crossover_benchmark.cpp it is an environment variable because the
// shared main() hands argv to Google Benchmark.

#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
// std::getenv is the portable call for the environment variable read below;
// MSVC warns about it as it does for every C standard call with a _s variant
// (benchmarks/tiered_memory_benchmark.cpp takes the same route).
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "benchmark_spread.h"
#include "crossover_circuit.h"
#include "cuda_probe_kernels.h"
#include "ir/gate_table.h"
#include "kernels/cuda/device_buffer.h"
#include "kernels/cuda/device_info.h"
#include "kernels/dense_kernels.h"
#include "menos/circuit.h"
#include "menos_benchmark_environment.h"

namespace {

using Clock = std::chrono::steady_clock;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;
using menos::kernels::DeviceState;
using menos::kernels::cuda::DeviceBuffer;
namespace probe = menos::benchmarks::cuda_probe;

// Repetitions per point, as in Phase 15 (owner, docs/adr/0019 Q1).
constexpr int kRepetitions = 10;

// Gates timed per sample in the per-gate breakdown (b).
constexpr int kGatesPerSample = 20;

constexpr std::uint32_t kMinQubits = 4;
constexpr std::uint32_t kMaxQubits = 26;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

nlohmann::json spread_json(const Spread& s) {
    return {{"median", s.median}, {"min", s.min}, {"max", s.max}, {"q1", s.q1}, {"q3", s.q3}};
}

std::uint64_t state_bytes(std::uint32_t num_qubits) {
    return std::uint64_t{16} << num_qubits;
}

// ---------------------------------------------------------------------------
// (a) the specialised kernels and the probe against the generic kernel
// ---------------------------------------------------------------------------

// One gate of the crossover circuit, resolved once outside the timed region
// for the probe path. The generic and specialised paths regenerate their
// matrix per gate, because CudaDenseEngine does.
struct ResolvedGate {
    menos::Opcode opcode;
    std::vector<menos::QubitId> qubits;
    std::vector<double> params;
};

std::vector<ResolvedGate> resolve(const menos::Circuit& circuit) {
    std::vector<ResolvedGate> gates;
    for (const menos::Operation& operation : circuit.operations()) {
        const auto* gate = std::get_if<menos::GateOp>(&operation);
        if (gate != nullptr) {
            gates.push_back({gate->opcode, gate->qubits, gate->params});
        }
    }
    return gates;
}

// CudaDenseEngine's apply_operations loop for GateOps, monitor-free: the
// shipped apply_matrix, which runs the specialised kernels for k = 1 and 2.
void run_specialised(DeviceState state, const std::vector<ResolvedGate>& gates) {
    for (const ResolvedGate& gate : gates) {
        const std::vector<std::complex<double>> matrix =
            menos::ir::generate_matrix(gate.opcode, gate.params);
        menos::kernels::cuda::apply_matrix(state, matrix, gate.qubits);
    }
}

// The same loop on the generic kernel for every gate: the path the engine took
// before Phase 16, and the oracle the gate's same-run ratios divide by.
void run_generic(DeviceState state, const std::vector<ResolvedGate>& gates) {
    for (const ResolvedGate& gate : gates) {
        const std::vector<std::complex<double>> matrix =
            menos::ir::generate_matrix(gate.opcode, gate.params);
        menos::kernels::cuda::apply_matrix_generic(state, matrix, gate.qubits);
    }
}

// The probe: H and CX on their hand-written kernels, anything else (T) on the
// generic kernel, as in the step-1 baseline.
void run_probe(DeviceState state, const std::vector<ResolvedGate>& gates) {
    for (const ResolvedGate& gate : gates) {
        if (gate.opcode == menos::Opcode::H) {
            probe::apply_h(state, gate.qubits[0]);
        } else if (gate.opcode == menos::Opcode::CX) {
            probe::apply_cx(state, gate.qubits[0], gate.qubits[1]);
        } else {
            const std::vector<std::complex<double>> matrix =
                menos::ir::generate_matrix(gate.opcode, gate.params);
            menos::kernels::cuda::apply_matrix_generic(state, matrix, gate.qubits);
        }
    }
}

using Runner = void (*)(DeviceState, const std::vector<ResolvedGate>&);

struct PathTiming {
    Spread gates_only;
    Spread end_to_end;
};

// One untimed warm-up, then kRepetitions timed runs. Each run allocates,
// initializes, applies every gate, copies the state into `host` and frees,
// as CudaDenseEngine::execute does for a statevector job; the gate loop is
// also timed on its own. `host` is allocated by the caller, outside the
// timing.
PathTiming time_path(Runner runner,
                     std::uint32_t num_qubits,
                     const std::vector<ResolvedGate>& gates,
                     std::span<std::complex<double>> host) {
    const std::size_t amplitudes = std::size_t{1} << num_qubits;
    const std::uint64_t budget = menos::kernels::cuda::device_available_bytes();
    std::vector<double> gates_only;
    std::vector<double> end_to_end;
    for (int r = -1; r < kRepetitions; ++r) {
        const Clock::time_point start = Clock::now();
        double gate_seconds = 0.0;
        {
            DeviceBuffer buffer(amplitudes, budget);
            menos::kernels::cuda::initialize(buffer.state(), 0.0);
            const Clock::time_point gates_start = Clock::now();
            runner(buffer.state(), gates);
            gate_seconds = seconds_since(gates_start);
            buffer.copy_to_host(host);
        }
        const double total = seconds_since(start);
        if (r >= 0) {
            gates_only.push_back(gate_seconds);
            end_to_end.push_back(total);
        }
    }
    return {spread(std::move(gates_only)), spread(std::move(end_to_end))};
}

double max_abs_difference(std::span<const std::complex<double>> a,
                          std::span<const std::complex<double>> b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

bool bytes_equal(std::span<const std::complex<double>> a, std::span<const std::complex<double>> b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(std::complex<double>)) == 0;
}

nlohmann::json probe_section() {
    std::printf(
        "\n== (a) the specialised kernels and the probe against the generic kernel, same run ==\n");
    std::printf("%3s %5s %11s %11s %7s %11s %7s %11s %11s %7s %11s %7s %7s %9s\n",
                "n",
                "gates",
                "generic(s)",
                "spec(s)",
                "g/spec",
                "probe(s)",
                "g/probe",
                "gen e2e(s)",
                "spec e2e(s)",
                "g/spec",
                "probe e2e",
                "g/probe",
                "bitwise",
                "probe|d|");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    for (std::uint32_t n = kMinQubits; n <= kMaxQubits; n += 2) {
        // Three host copies of the state (one per path) must fit the host
        // budget, and one device state the device.
        if (state_bytes(n) > device_budget || 3 * state_bytes(n) > host_budget) {
            std::printf("%3u (skipped: does not fit the device or host budget)\n", n);
            rows.push_back({{"n", n}, {"measured", false}});
            continue;
        }
        const std::vector<ResolvedGate> gates =
            resolve(menos::benchmarks::make_crossover_circuit(n));
        std::vector<std::complex<double>> generic_host(std::size_t{1} << n);
        std::vector<std::complex<double>> specialised_host(std::size_t{1} << n);
        std::vector<std::complex<double>> probe_host(std::size_t{1} << n);
        const PathTiming generic = time_path(&run_generic, n, gates, generic_host);
        const PathTiming specialised = time_path(&run_specialised, n, gates, specialised_host);
        const PathTiming probed = time_path(&run_probe, n, gates, probe_host);
        const bool bitwise = bytes_equal(generic_host, specialised_host);
        const double difference = max_abs_difference(generic_host, probe_host);
        const double specialised_ratio = generic.gates_only.median / specialised.gates_only.median;
        const double specialised_e2e_ratio =
            generic.end_to_end.median / specialised.end_to_end.median;
        const double ratio = generic.gates_only.median / probed.gates_only.median;
        const double e2e_ratio = generic.end_to_end.median / probed.end_to_end.median;
        std::printf(
            "%3u %5zu %11.6f %11.6f %6.2fx %11.6f %6.2fx %11.6f %11.6f %6.2fx %11.6f %6.2fx %7s "
            "%9.2e\n",
            n,
            gates.size(),
            generic.gates_only.median,
            specialised.gates_only.median,
            specialised_ratio,
            probed.gates_only.median,
            ratio,
            generic.end_to_end.median,
            specialised.end_to_end.median,
            specialised_e2e_ratio,
            probed.end_to_end.median,
            e2e_ratio,
            bitwise ? "yes" : "NO",
            difference);
        std::fflush(stdout);
        rows.push_back({
            {"n", n},
            {"measured", true},
            {"gates", gates.size()},
            {"generic_gates_seconds", spread_json(generic.gates_only)},
            {"specialised_gates_seconds", spread_json(specialised.gates_only)},
            {"probe_gates_seconds", spread_json(probed.gates_only)},
            {"generic_end_to_end_seconds", spread_json(generic.end_to_end)},
            {"specialised_end_to_end_seconds", spread_json(specialised.end_to_end)},
            {"probe_end_to_end_seconds", spread_json(probed.end_to_end)},
            {"specialised_gates_ratio", specialised_ratio},
            {"specialised_end_to_end_ratio", specialised_e2e_ratio},
            {"specialised_bitwise_identical", bitwise},
            {"gates_ratio", ratio},
            {"end_to_end_ratio", e2e_ratio},
            {"max_abs_difference", difference},
        });
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (b) one gate's cost in parts
// ---------------------------------------------------------------------------

// kRepetitions samples, each the mean over kGatesPerSample calls of `call`.
Spread time_host(const std::function<void()>& call) {
    call();  // warm
    std::vector<double> samples;
    for (int r = 0; r < kRepetitions; ++r) {
        const Clock::time_point start = Clock::now();
        for (int g = 0; g < kGatesPerSample; ++g) {
            call();
        }
        samples.push_back(seconds_since(start) / kGatesPerSample);
    }
    return spread(std::move(samples));
}

// kRepetitions samples, each the mean CUDA-event span of one call over
// kGatesPerSample calls.
Spread time_device(const std::function<void()>& call) {
    call();  // warm
    std::vector<double> samples;
    for (int r = 0; r < kRepetitions; ++r) {
        double total = 0.0;
        for (int g = 0; g < kGatesPerSample; ++g) {
            total += probe::device_seconds(call);
        }
        samples.push_back(total / kGatesPerSample);
    }
    return spread(std::move(samples));
}

nlohmann::json breakdown_section() {
    std::printf("\n== (b) one gate's cost in parts (microseconds, medians) ==\n");
    std::printf("%3s %3s %13s %13s %13s %13s %8s %13s %13s %13s %14s\n",
                "n",
                "k",
                "generic wall",
                "generic event",
                "spec wall",
                "spec event",
                "g/spec",
                "probe wall",
                "probe event",
                "launch+sync",
                "2 const copies");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    for (const std::uint32_t n : {4U, 8U, 12U, 16U, 20U, 22U, 24U, 26U}) {
        if (state_bytes(n) > device_budget) {
            std::printf("%3u (skipped: does not fit the device)\n", n);
            continue;
        }
        DeviceBuffer buffer(std::size_t{1} << n, device_budget);
        menos::kernels::cuda::initialize(buffer.state(), 0.0);
        const DeviceState state = buffer.state();
        // A middle qubit, so neither the lowest nor the highest stride is
        // favoured; CX on the pair (q, q + 1) as in the circuit family.
        const std::uint32_t q = n / 2;
        for (const int k : {1, 2}) {
            const menos::Opcode opcode = k == 1 ? menos::Opcode::H : menos::Opcode::CX;
            const std::vector<std::uint32_t> operands =
                k == 1 ? std::vector<std::uint32_t>{q} : std::vector<std::uint32_t>{q, q + 1};
            const std::vector<std::complex<double>> matrix = menos::ir::generate_matrix(opcode, {});
            const auto generic = [&] {
                menos::kernels::cuda::apply_matrix_generic(state, matrix, operands);
            };
            const auto specialised = [&] {
                menos::kernels::cuda::apply_matrix(state, matrix, operands);
            };
            const auto probed = [&] {
                if (k == 1) {
                    probe::apply_h(state, q);
                } else {
                    probe::apply_cx(state, q, q + 1);
                }
            };
            const auto copies = [&] {
                probe::copy_gate_to_constant(
                    matrix.data(), matrix.size(), operands.data(), operands.size());
            };
            const Spread generic_wall = time_host(generic);
            const Spread generic_event = time_device(generic);
            const Spread specialised_wall = time_host(specialised);
            const Spread specialised_event = time_device(specialised);
            const double ratio = generic_wall.median / specialised_wall.median;
            const Spread probe_wall = time_host(probed);
            const Spread probe_event = time_device(probed);
            const Spread copy_wall = time_host(copies);
            // Measured beside each row, not once: the GPU's clocks move with
            // load, and a figure taken at a different moment is not
            // comparable with the row's own.
            const Spread launch = time_host(&probe::empty_launch_and_sync);
            constexpr double kMicro = 1e6;
            std::printf("%3u %3d %13.2f %13.2f %13.2f %13.2f %7.2fx %13.2f %13.2f %13.2f %14.2f\n",
                        n,
                        k,
                        generic_wall.median * kMicro,
                        generic_event.median * kMicro,
                        specialised_wall.median * kMicro,
                        specialised_event.median * kMicro,
                        ratio,
                        probe_wall.median * kMicro,
                        probe_event.median * kMicro,
                        launch.median * kMicro,
                        copy_wall.median * kMicro);
            std::fflush(stdout);
            rows.push_back({
                {"n", n},
                {"k", k},
                {"operands", operands},
                {"generic_wall_seconds", spread_json(generic_wall)},
                {"generic_event_seconds", spread_json(generic_event)},
                {"specialised_wall_seconds", spread_json(specialised_wall)},
                {"specialised_event_seconds", spread_json(specialised_event)},
                {"specialised_wall_ratio", ratio},
                {"probe_wall_seconds", spread_json(probe_wall)},
                {"probe_event_seconds", spread_json(probe_event)},
                {"constant_copies_wall_seconds", spread_json(copy_wall)},
                {"launch_and_sync_seconds", spread_json(launch)},
            });
        }
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (c) bandwidth
// ---------------------------------------------------------------------------

nlohmann::json bandwidth_section() {
    std::printf("\n== (c) device memory bandwidth (read + write bytes / time) ==\n");
    std::printf(
        "%3s %14s %14s %14s %14s\n", "n", "copy (s)", "copy GB/s", "touch (s)", "touch GB/s");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    for (const std::uint32_t n : {20U, 22U, 24U, 26U}) {
        if (2 * state_bytes(n) > device_budget) {
            std::printf("%3u (skipped: two states do not fit the device)\n", n);
            continue;
        }
        const std::size_t amplitudes = std::size_t{1} << n;
        DeviceBuffer source(amplitudes, device_budget);
        DeviceBuffer destination(amplitudes, device_budget);
        const std::size_t bytes = state_bytes(n);
        const auto copy = [&] {
            probe::copy_device_to_device(destination.state().data, source.state().data, bytes);
        };
        const auto touch = [&] { probe::touch_every_amplitude(destination.state()); };
        std::vector<double> copy_samples;
        std::vector<double> touch_samples;
        copy();
        touch();
        for (int r = 0; r < kRepetitions; ++r) {
            copy_samples.push_back(probe::device_seconds(copy));
            touch_samples.push_back(probe::device_seconds(touch));
        }
        const Spread copy_time = spread(std::move(copy_samples));
        const Spread touch_time = spread(std::move(touch_samples));
        const double moved = 2.0 * static_cast<double>(bytes);
        constexpr double kGiga = 1e9;
        std::printf("%3u %14.6f %14.1f %14.6f %14.1f\n",
                    n,
                    copy_time.median,
                    moved / copy_time.median / kGiga,
                    touch_time.median,
                    moved / touch_time.median / kGiga);
        std::fflush(stdout);
        rows.push_back({
            {"n", n},
            {"bytes_moved", moved},
            {"copy_seconds", spread_json(copy_time)},
            {"touch_seconds", spread_json(touch_time)},
            {"copy_gb_per_second", moved / copy_time.median / kGiga},
            {"touch_gb_per_second", moved / touch_time.median / kGiga},
        });
    }
    return rows;
}

// About a second of full-bandwidth work before anything is timed, so the first
// rows are not measured while the GPU is still raising its clocks from idle
// (the first draft of this benchmark showed it: its n = 4 rows were the
// slowest small-n rows).
void warm_up_gpu() {
    constexpr std::uint32_t kWarmQubits = 24;
    constexpr int kWarmPasses = 250;
    DeviceBuffer buffer(std::size_t{1} << kWarmQubits,
                        menos::kernels::cuda::device_available_bytes());
    for (int pass = 0; pass < kWarmPasses; ++pass) {
        probe::touch_every_amplitude(buffer.state());
    }
}

void write_json(const nlohmann::json& document) {
    const char* const path = std::getenv("MENOS_CUDA_KERNEL_JSON");
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
    std::printf("\n=== MENOS CUDA kernel benchmark (Phase 16) ===\n");
    nlohmann::json document = {{"schema_version", 2}, {"device_present", info.present}};
    if (!info.present) {
        std::printf("no CUDA device available -- nothing measured\n");
        write_json(document);
        return;
    }
    std::printf("environment      : %s\n", menos_benchmark::describe_environment().c_str());
    std::printf("GPU              : %s (compute capability %d.%d, %d SMs)\n",
                info.name.c_str(),
                info.compute_capability_major,
                info.compute_capability_minor,
                info.multiprocessor_count);
    std::printf("repetitions      : %d per point after a warm-up; medians unless stated\n",
                kRepetitions);
    document["environment"] = menos_benchmark::describe_environment();
    document["gpu"] = {
        {"name", info.name},
        {"compute_capability",
         std::to_string(info.compute_capability_major) + "." +
             std::to_string(info.compute_capability_minor)},
        {"multiprocessors", info.multiprocessor_count},
        {"total_memory_bytes", info.total_memory_bytes},
        {"free_memory_bytes", info.free_memory_bytes},
    };
    document["repetitions"] = kRepetitions;
    document["gates_per_sample"] = kGatesPerSample;
    document["circuit"] = {
        {"layers", menos::benchmarks::kCrossoverLayers},
        {"description", "H on all qubits, T on q0, CX on alternating pairs, per layer"},
    };
    warm_up_gpu();
    document["probe"] = probe_section();
    document["breakdown"] = breakdown_section();
    document["bandwidth"] = bandwidth_section();
    std::printf("\n");
    write_json(document);
}

}  // namespace

// Report-only, like BM_CudaCrossoverReport.
static void BM_CudaKernelReport(benchmark::State& state) {
    for (auto _ : state) {
        report();
        state.SkipWithError("report-only benchmark: see the printed tables");
        break;
    }
}
BENCHMARK(BM_CudaKernelReport)->Iterations(1)->Unit(benchmark::kSecond);
