// The Phase 23 step-1 tiered-memory benchmark, CUDA side (docs/adr/0027
// sections 4 and 6, step 1): what the host-offloaded tier would pay per
// chunk, what unified memory does under this driver, and the in-VRAM engine
// it would compete with -- measured before any engine code exists, all in
// one run because the GPU's clock state varies about 2x between runs
// (docs/adr/0020 section 10.4).
//
//   (0) facts: the device, its free memory, and the managed-memory attributes
//       the driver reports;
//   (1) transfers: host-to-device and device-to-host of a 1 GiB and a 2 GiB
//       chunk from pageable memory (a std::vector, as copy_to_host uses
//       today), from a cudaHostAlloc buffer and from a cudaHostRegister'ed
//       vector -- Phase 22 measured these at up to 64 MiB and found no gain
//       for page-locking (docs/adr/0026 section 10.1); chunk-sized transfers
//       are what section 3.4 would make;
//   (2) windows: copy a chunk in, apply 1, 10 and 25 one-qubit gates with the
//       shipped kernels, copy it out -- the per-pass cost of section 3.1 at
//       chunk scale -- against the same gates on the resident chunk;
//   (3) overlap: whether a copy on a second stream proceeds while the device
//       is busy, measured with cudaMemsetAsync as the work (the shipped
//       kernels synchronise the whole device after every launch, so they
//       cannot overlap anything; this row says what lifting that would buy);
//   (4) managed: cudaMallocManaged at 2^29 and 2^30 amplitudes, beyond this
//       device's memory, with a one-qubit gate sweep by the shipped kernel:
//       whether it allocates, whether the sweep completes, and its time;
//   (5) in VRAM: CudaDenseEngine::execute on the crossover circuit and on an
//       empty circuit at n = 26-28, for the per-gate cost of the competitor.
//
// This is a benchmark, not a test (CLAUDE.md): it asserts nothing. Setting
// MENOS_CUDA_TIERED_JSON=<path> also writes every table as JSON, for
// docs/benchmarks/phase-23-baseline/. It calls the CUDA runtime directly, as
// cuda_memory_benchmark.cpp does.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "benchmark_spread.h"
#include "crossover_circuit.h"
#include "engines/dense_cuda/cuda_dense_engine.h"
#include "ir/gate_table.h"
#include "kernels/cuda/cuda_error.h"
#include "kernels/cuda/device_buffer.h"
#include "kernels/cuda/device_info.h"
#include "kernels/dense_kernels.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"
#include "menos_benchmark_environment.h"

namespace {

using Clock = std::chrono::steady_clock;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;
using menos::kernels::cuda::DeviceBuffer;
using menos::kernels::cuda::ensure_cuda;

constexpr int kRepetitions = 10;
constexpr int kLargeRepetitions = 3;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
constexpr std::uint64_t kChunkSizes[] = {1 * kGiB, 2 * kGiB};
constexpr int kGateCounts[] = {1, 10, 25};
constexpr std::uint32_t kManagedQubits[] = {29, 30};
constexpr std::uint32_t kMinInVramQubits = 26;
constexpr std::uint32_t kMaxInVramQubits = 28;
constexpr double kGiga = 1e9;
constexpr double kMilli = 1e3;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

nlohmann::json spread_json(const Spread& s) {
    return {{"median", s.median}, {"min", s.min}, {"max", s.max}, {"q1", s.q1}, {"q3", s.q3}};
}

double gb_per_second(std::uint64_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / kGiga / seconds : 0.0;
}

// kRepetitions samples of one call each, after one untimed call.
Spread time_host(const std::function<void()>& call, int repetitions) {
    call();
    std::vector<double> samples;
    for (int r = 0; r < repetitions; ++r) {
        const Clock::time_point start = Clock::now();
        call();
        samples.push_back(seconds_since(start));
    }
    return spread(std::move(samples));
}

std::size_t amplitudes_of(std::uint64_t bytes) {
    return static_cast<std::size_t>(bytes / sizeof(std::complex<double>));
}

// A page-locked host buffer from cudaHostAlloc.
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
        ensure_cuda(cudaHostAlloc(&data_, bytes, cudaHostAllocDefault), "cudaHostAlloc");
        std::fill_n(static_cast<unsigned char*>(data_), bytes, static_cast<unsigned char>(1));
    }
    ~PinnedBuffer() {
        static_cast<void>(cudaFreeHost(data_));
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&&) = delete;
    PinnedBuffer& operator=(PinnedBuffer&&) = delete;
    [[nodiscard]] void* data() const {
        return data_;
    }
    [[nodiscard]] std::size_t bytes() const {
        return bytes_;
    }

private:
    void* data_ = nullptr;
    std::size_t bytes_;
};

// A pageable vector page-locked in place for its lifetime.
class RegisteredVector {
public:
    explicit RegisteredVector(std::size_t amplitudes)
        : data_(amplitudes, std::complex<double>{1.0, 0.0}) {
        ensure_cuda(
            cudaHostRegister(
                data_.data(), data_.size() * sizeof(std::complex<double>), cudaHostRegisterDefault),
            "cudaHostRegister");
    }
    ~RegisteredVector() {
        static_cast<void>(cudaHostUnregister(data_.data()));
    }
    RegisteredVector(const RegisteredVector&) = delete;
    RegisteredVector& operator=(const RegisteredVector&) = delete;
    RegisteredVector(RegisteredVector&&) = delete;
    RegisteredVector& operator=(RegisteredVector&&) = delete;
    [[nodiscard]] void* data() {
        return data_.data();
    }

private:
    std::vector<std::complex<double>> data_;
};

// ---------------------------------------------------------------------------
// (0) facts
// ---------------------------------------------------------------------------

struct Facts {
    int managed_memory = 0;
    int concurrent_managed_access = 0;
    int pageable_memory_access = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

Facts query_facts() {
    Facts facts;
    ensure_cuda(cudaRuntimeGetVersion(&facts.runtime_version), "cudaRuntimeGetVersion");
    ensure_cuda(cudaDriverGetVersion(&facts.driver_version), "cudaDriverGetVersion");
    ensure_cuda(cudaDeviceGetAttribute(&facts.managed_memory, cudaDevAttrManagedMemory, 0),
                "cudaDeviceGetAttribute (ManagedMemory)");
    ensure_cuda(cudaDeviceGetAttribute(
                    &facts.concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, 0),
                "cudaDeviceGetAttribute (ConcurrentManagedAccess)");
    ensure_cuda(
        cudaDeviceGetAttribute(&facts.pageable_memory_access, cudaDevAttrPageableMemoryAccess, 0),
        "cudaDeviceGetAttribute (PageableMemoryAccess)");
    return facts;
}

nlohmann::json facts_section(const Facts& facts) {
    std::printf("\n== (0) facts ==\n");
    std::printf(
        "runtime / driver version : %d / %d\n", facts.runtime_version, facts.driver_version);
    std::printf("managed memory           : %d\n", facts.managed_memory);
    std::printf("concurrent managed access: %d\n", facts.concurrent_managed_access);
    std::printf("pageable memory access   : %d\n", facts.pageable_memory_access);
    return {
        {"runtime_version", facts.runtime_version},
        {"driver_version", facts.driver_version},
        {"managed_memory", facts.managed_memory},
        {"concurrent_managed_access", facts.concurrent_managed_access},
        {"pageable_memory_access", facts.pageable_memory_access},
    };
}

// ---------------------------------------------------------------------------
// (1) transfers
// ---------------------------------------------------------------------------

nlohmann::json transfer_section() {
    std::printf("\n== (1) chunk transfers (milliseconds, medians; GB/s of the median) ==\n");
    std::printf("%5s %-10s %10s %7s %10s %7s\n", "GiB", "source", "H2D", "GB/s", "D2H", "GB/s");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    for (const std::uint64_t bytes : kChunkSizes) {
        if (bytes > device_budget || 2 * bytes > host_budget) {
            std::printf("%5llu (skipped: does not fit the device or host budget)\n",
                        static_cast<unsigned long long>(bytes / kGiB));
            rows.push_back({{"chunk_bytes", bytes}, {"measured", false}});
            continue;
        }
        const std::size_t amplitudes = amplitudes_of(bytes);
        DeviceBuffer device(amplitudes, device_budget);
        menos::kernels::cuda::initialize(device.state(), 0.0);
        const auto copy_in = [&](const void* source) {
            ensure_cuda(cudaMemcpy(device.state().data, source, bytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy (H2D)");
        };
        const auto copy_out = [&](void* destination) {
            ensure_cuda(cudaMemcpy(destination, device.state().data, bytes, cudaMemcpyDeviceToHost),
                        "cudaMemcpy (D2H)");
        };
        struct Source {
            const char* name;
            Spread h2d;
            Spread d2h;
        };
        std::vector<Source> sources;
        {
            std::vector<std::complex<double>> pageable(amplitudes, std::complex<double>{1.0, 0.0});
            sources.push_back({"pageable",
                               time_host([&] { copy_in(pageable.data()); }, kRepetitions),
                               time_host([&] { copy_out(pageable.data()); }, kRepetitions)});
        }
        {
            const PinnedBuffer pinned(static_cast<std::size_t>(bytes));
            sources.push_back({"pinned",
                               time_host([&] { copy_in(pinned.data()); }, kRepetitions),
                               time_host([&] { copy_out(pinned.data()); }, kRepetitions)});
        }
        {
            RegisteredVector registered(amplitudes);
            sources.push_back({"registered",
                               time_host([&] { copy_in(registered.data()); }, kRepetitions),
                               time_host([&] { copy_out(registered.data()); }, kRepetitions)});
        }
        for (const Source& source : sources) {
            std::printf("%5llu %-10s %10.2f %7.2f %10.2f %7.2f\n",
                        static_cast<unsigned long long>(bytes / kGiB),
                        source.name,
                        source.h2d.median * kMilli,
                        gb_per_second(bytes, source.h2d.median),
                        source.d2h.median * kMilli,
                        gb_per_second(bytes, source.d2h.median));
            rows.push_back({
                {"chunk_bytes", bytes},
                {"measured", true},
                {"source", source.name},
                {"h2d_seconds", spread_json(source.h2d)},
                {"d2h_seconds", spread_json(source.d2h)},
                {"h2d_gb_per_second", gb_per_second(bytes, source.h2d.median)},
                {"d2h_gb_per_second", gb_per_second(bytes, source.d2h.median)},
            });
        }
        std::fflush(stdout);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (2) windows
// ---------------------------------------------------------------------------

// `count` one-qubit gates on alternating low qubits of the resident window,
// through the shipped kernel (one synchronisation per gate, as the engine).
// count (how many gates) and window_qubits (a width) are passed from named
// values at the two call sites and are not confusable there.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void apply_gates(menos::kernels::DeviceState window, int count, std::uint32_t window_qubits) {
    const std::vector<std::complex<double>> hadamard =
        menos::ir::generate_matrix(menos::Opcode::H, {});
    for (int g = 0; g < count; ++g) {
        const std::uint32_t qubit = static_cast<std::uint32_t>(g) % window_qubits;
        const std::vector<std::uint32_t> operands{qubit};
        menos::kernels::cuda::apply_matrix(window, hadamard, operands);
    }
}

nlohmann::json window_section() {
    std::printf(
        "\n== (2) a window copied in, k one-qubit gates, copied out (milliseconds, medians) ==\n");
    std::printf("%5s %3s %-10s %10s %10s %10s %8s\n",
                "GiB",
                "k",
                "source",
                "in+k+out",
                "k alone",
                "copies",
                "cp/gate");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    for (const std::uint64_t bytes : kChunkSizes) {
        if (bytes > device_budget || 2 * bytes > host_budget) {
            std::printf("%5llu (skipped: does not fit the device or host budget)\n",
                        static_cast<unsigned long long>(bytes / kGiB));
            rows.push_back({{"chunk_bytes", bytes}, {"measured", false}});
            continue;
        }
        const std::size_t amplitudes = amplitudes_of(bytes);
        std::uint32_t window_qubits = 0;
        while ((std::size_t{1} << window_qubits) < amplitudes) {
            ++window_qubits;
        }
        DeviceBuffer device(amplitudes, device_budget);
        menos::kernels::cuda::initialize(device.state(), 0.0);
        std::vector<std::complex<double>> pageable(amplitudes, std::complex<double>{1.0, 0.0});
        const PinnedBuffer pinned(static_cast<std::size_t>(bytes));
        const std::pair<const char*, void*> sources[] = {{"pageable", pageable.data()},
                                                         {"pinned", pinned.data()}};
        for (const int count : kGateCounts) {
            const Spread alone =
                time_host([&] { apply_gates(device.state(), count, window_qubits); }, kRepetitions);
            for (const auto& [name, host] : sources) {
                const Spread pass = time_host(
                    [&] {
                        ensure_cuda(
                            cudaMemcpy(device.state().data, host, bytes, cudaMemcpyHostToDevice),
                            "cudaMemcpy (window in)");
                        apply_gates(device.state(), count, window_qubits);
                        ensure_cuda(
                            cudaMemcpy(host, device.state().data, bytes, cudaMemcpyDeviceToHost),
                            "cudaMemcpy (window out)");
                    },
                    kRepetitions);
                const double copies = pass.median - alone.median;
                std::printf("%5llu %3d %-10s %10.2f %10.2f %10.2f %8.2f\n",
                            static_cast<unsigned long long>(bytes / kGiB),
                            count,
                            name,
                            pass.median * kMilli,
                            alone.median * kMilli,
                            copies * kMilli,
                            copies / (alone.median / count));
                rows.push_back({
                    {"chunk_bytes", bytes},
                    {"measured", true},
                    {"gates", count},
                    {"source", name},
                    {"pass_seconds", spread_json(pass)},
                    {"gates_alone_seconds", spread_json(alone)},
                    {"copies_seconds", copies},
                    {"copies_in_gates", copies / (alone.median / count)},
                });
            }
        }
        std::fflush(stdout);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (3) overlap
// ---------------------------------------------------------------------------

nlohmann::json overlap_section() {
    std::printf(
        "\n== (3) a pinned copy on a second stream against memsets on the first "
        "(milliseconds, medians) ==\n");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    const std::uint64_t bytes = kChunkSizes[0];
    if (2 * bytes > device_budget || bytes > host_budget) {
        std::printf("(skipped: two chunks do not fit the device)\n");
        return rows;
    }
    const std::size_t amplitudes = amplitudes_of(bytes);
    DeviceBuffer work(amplitudes, device_budget);
    DeviceBuffer target(amplitudes, device_budget);
    const PinnedBuffer pinned(static_cast<std::size_t>(bytes));
    cudaStream_t copy_stream = nullptr;
    cudaStream_t work_stream = nullptr;
    ensure_cuda(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    ensure_cuda(cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    constexpr int kMemsets = 8;
    const auto memsets = [&] {
        for (int i = 0; i < kMemsets; ++i) {
            ensure_cuda(cudaMemsetAsync(work.state().data, i & 1, bytes, work_stream),
                        "cudaMemsetAsync");
        }
    };
    const auto copy = [&] {
        ensure_cuda(
            cudaMemcpyAsync(
                target.state().data, pinned.data(), bytes, cudaMemcpyHostToDevice, copy_stream),
            "cudaMemcpyAsync");
    };
    const auto wait = [&] { ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); };
    const Spread work_only = time_host(
        [&] {
            memsets();
            wait();
        },
        kRepetitions);
    const Spread copy_only = time_host(
        [&] {
            copy();
            wait();
        },
        kRepetitions);
    const Spread sequential = time_host(
        [&] {
            memsets();
            wait();
            copy();
            wait();
        },
        kRepetitions);
    const Spread concurrent = time_host(
        [&] {
            memsets();
            copy();
            wait();
        },
        kRepetitions);
    ensure_cuda(cudaStreamDestroy(copy_stream), "cudaStreamDestroy");
    ensure_cuda(cudaStreamDestroy(work_stream), "cudaStreamDestroy");
    std::printf("%-14s %10.2f\n", "work only", work_only.median * kMilli);
    std::printf("%-14s %10.2f\n", "copy only", copy_only.median * kMilli);
    std::printf("%-14s %10.2f\n", "sequential", sequential.median * kMilli);
    std::printf("%-14s %10.2f\n", "concurrent", concurrent.median * kMilli);
    std::printf("%-14s %10.2f\n", "overlap gain", (sequential.median - concurrent.median) * kMilli);
    std::fflush(stdout);
    rows.push_back({
        {"chunk_bytes", bytes},
        {"memsets", kMemsets},
        {"work_only_seconds", spread_json(work_only)},
        {"copy_only_seconds", spread_json(copy_only)},
        {"sequential_seconds", spread_json(sequential)},
        {"concurrent_seconds", spread_json(concurrent)},
        {"overlap_gain_seconds", sequential.median - concurrent.median},
    });
    return rows;
}

// ---------------------------------------------------------------------------
// (4) managed
// ---------------------------------------------------------------------------

nlohmann::json managed_section(const Facts& facts) {
    std::printf(
        "\n== (4) cudaMallocManaged beyond device memory, one-qubit sweeps (seconds, medians) "
        "==\n");
    std::printf("%3s %12s %-12s %10s %10s\n", "n", "bytes", "outcome", "sweep", "resident/2^28");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    const std::vector<std::complex<double>> hadamard =
        menos::ir::generate_matrix(menos::Opcode::H, {});
    const std::vector<std::uint32_t> operands{3};
    // The resident reference: the same sweep on the largest state that fits.
    constexpr std::uint32_t kResidentQubits = 28;
    constexpr std::uint64_t kResidentBytes = std::uint64_t{16} << kResidentQubits;
    Spread resident{};
    bool resident_measured = false;
    if (kResidentBytes <= device_budget) {
        DeviceBuffer buffer(std::size_t{1} << kResidentQubits, device_budget);
        menos::kernels::cuda::initialize(buffer.state(), 0.0);
        resident = time_host(
            [&] { menos::kernels::cuda::apply_matrix(buffer.state(), hadamard, operands); },
            kRepetitions);
        resident_measured = true;
        std::printf("%3u %12llu %-12s %10.4f (resident reference)\n",
                    kResidentQubits,
                    static_cast<unsigned long long>(kResidentBytes),
                    "resident",
                    resident.median);
    }
    for (const std::uint32_t n : kManagedQubits) {
        const std::uint64_t bytes = std::uint64_t{16} << n;
        nlohmann::json row = {{"n", n}, {"bytes", bytes}, {"measured", false}};
        if (facts.managed_memory == 0) {
            std::printf(
                "%3u %12llu %-12s\n", n, static_cast<unsigned long long>(bytes), "unsupported");
            row["outcome"] = "unsupported";
            rows.push_back(row);
            continue;
        }
        if (bytes > host_budget) {
            std::printf(
                "%3u %12llu %-12s\n", n, static_cast<unsigned long long>(bytes), "host budget");
            row["outcome"] = "exceeds host budget";
            rows.push_back(row);
            continue;
        }
        void* managed = nullptr;
        const cudaError_t allocated = cudaMallocManaged(&managed, bytes, cudaMemAttachGlobal);
        if (allocated != cudaSuccess) {
            std::printf("%3u %12llu %-12s %s\n",
                        n,
                        static_cast<unsigned long long>(bytes),
                        "alloc failed",
                        cudaGetErrorString(allocated));
            row["outcome"] = std::string("allocation failed: ") + cudaGetErrorString(allocated);
            static_cast<void>(cudaGetLastError());
            rows.push_back(row);
            continue;
        }
        const menos::kernels::DeviceState state{managed, std::size_t{1} << n};
        std::string outcome = "ok";
        Spread sweep{};
        try {
            menos::kernels::cuda::initialize(state, 0.0);
            sweep =
                time_host([&] { menos::kernels::cuda::apply_matrix(state, hadamard, operands); },
                          kLargeRepetitions);
        } catch (const std::exception& error) {
            outcome = std::string("sweep failed: ") + error.what();
            static_cast<void>(cudaGetLastError());
        }
        static_cast<void>(cudaFree(managed));
        const double scaled_resident =
            resident_measured
                ? resident.median * static_cast<double>(std::uint64_t{1} << (n - kResidentQubits))
                : 0.0;
        std::printf("%3u %12llu %-12s %10.4f %10.4f\n",
                    n,
                    static_cast<unsigned long long>(bytes),
                    outcome == "ok" ? "ok" : "failed",
                    sweep.median,
                    scaled_resident);
        if (outcome != "ok") {
            std::printf("    %s\n", outcome.c_str());
        }
        row["measured"] = outcome == "ok";
        row["outcome"] = outcome;
        row["sweep_seconds"] = spread_json(sweep);
        row["resident_sweep_scaled_seconds"] = scaled_resident;
        rows.push_back(row);
        std::fflush(stdout);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (5) in VRAM
// ---------------------------------------------------------------------------

Spread time_execute(const menos::engines::dense_cuda::CudaDenseEngine& engine,
                    const menos::EngineExecutionRequest& request,
                    int repetitions) {
    std::vector<double> samples;
    for (int r = -1; r < repetitions; ++r) {
        const Clock::time_point start = Clock::now();
        const menos::EngineExecutionResult result = engine.execute(request);
        const double seconds = seconds_since(start);
        std::uint64_t actual_bytes = result.actual_bytes;
        benchmark::DoNotOptimize(actual_bytes);
        if (r >= 0) {
            samples.push_back(seconds);
        }
    }
    return spread(std::move(samples));
}

nlohmann::json in_vram_section() {
    std::printf(
        "\n== (5) CudaDenseEngine::execute in VRAM, crossover circuit and an empty circuit "
        "(seconds, medians; expectation output, no copy back) ==\n");
    std::printf("%3s %5s %11s %11s %11s\n", "n", "gates", "circuit", "empty", "per gate");
    nlohmann::json rows = nlohmann::json::array();
    const menos::engines::dense_cuda::CudaDenseEngine engine;
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    for (std::uint32_t n = kMinInVramQubits; n <= kMaxInVramQubits; ++n) {
        const std::uint64_t bytes = std::uint64_t{16} << n;
        if (bytes + menos::kernels::cuda::expectation_scratch_bytes() > device_budget) {
            std::printf("%3u (skipped: does not fit the device)\n", n);
            rows.push_back({{"n", n}, {"measured", false}});
            continue;
        }
        menos::Observable one;
        {
            menos::PauliString pauli;
            pauli.num_qubits = n;
            pauli.z_words = {std::uint64_t{1}};
            pauli.x_words = {std::uint64_t{0}};
            one.terms.emplace_back(pauli, std::complex<double>{1.0, 0.0});
        }
        const menos::Circuit circuit = menos::benchmarks::make_crossover_circuit(n);
        const menos::Circuit empty(n, 0);
        const auto request = [&](const menos::Circuit& c) {
            return menos::EngineExecutionRequest{
                .num_qubits = n,
                .operations = c.operations(),
                .want_statevector = false,
                .want_counts = false,
                .want_expectation = true,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = 0,
                .global_phase = 0.0,
                .budget_bytes = host_budget,
                .is_sole_region = true,
                .observable = &one,
            };
        };
        const Spread full = time_execute(engine, request(circuit), kRepetitions);
        const Spread init_only = time_execute(engine, request(empty), kRepetitions);
        const double per_gate =
            (full.median - init_only.median) / static_cast<double>(circuit.operations().size());
        std::printf("%3u %5zu %11.4f %11.4f %11.6f\n",
                    n,
                    circuit.operations().size(),
                    full.median,
                    init_only.median,
                    per_gate);
        std::fflush(stdout);
        rows.push_back({
            {"n", n},
            {"measured", true},
            {"gates", circuit.operations().size()},
            {"circuit_seconds", spread_json(full)},
            {"empty_circuit_seconds", spread_json(init_only)},
            {"per_gate_seconds", per_gate},
        });
    }
    return rows;
}

void warm_up_gpu() {
    constexpr std::uint32_t kWarmQubits = 24;
    constexpr int kWarmPasses = 100;
    DeviceBuffer buffer(std::size_t{1} << kWarmQubits,
                        menos::kernels::cuda::device_available_bytes());
    const std::size_t bytes = std::size_t{16} << kWarmQubits;
    for (int pass = 0; pass < kWarmPasses; ++pass) {
        ensure_cuda(cudaMemset(buffer.state().data, pass & 1, bytes), "cudaMemset (warm-up)");
    }
    ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warm-up)");
}

void write_json(const nlohmann::json& document) {
    const char* const path = std::getenv("MENOS_CUDA_TIERED_JSON");
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
    std::printf("\n=== MENOS tiered memory benchmark, CUDA side (Phase 23 step 1) ===\n");
    nlohmann::json document = {{"schema_version", 1}, {"device_present", info.present}};
    if (!info.present) {
        std::printf("no CUDA device available -- nothing measured\n");
        write_json(document);
        return;
    }
    std::printf("environment      : %s\n", menos_benchmark::describe_environment().c_str());
    std::printf(
        "GPU              : %s (compute capability %d.%d, %d SMs, %llu bytes total, %llu free)\n",
        info.name.c_str(),
        info.compute_capability_major,
        info.compute_capability_minor,
        info.multiprocessor_count,
        static_cast<unsigned long long>(info.total_memory_bytes),
        static_cast<unsigned long long>(menos::kernels::cuda::device_free_bytes()));
    std::printf(
        "repetitions      : %d per point (%d for managed sweeps) after a warm-up; medians\n",
        kRepetitions,
        kLargeRepetitions);
    document["environment"] = menos_benchmark::describe_environment();
    document["gpu"] = {
        {"name", info.name},
        {"compute_capability",
         std::to_string(info.compute_capability_major) + "." +
             std::to_string(info.compute_capability_minor)},
        {"multiprocessors", info.multiprocessor_count},
        {"total_memory_bytes", info.total_memory_bytes},
        {"free_memory_bytes", menos::kernels::cuda::device_free_bytes()},
    };
    document["repetitions"] = kRepetitions;
    document["host_budget_bytes"] = menos_benchmark::safe_allocation_budget_bytes();
    const Facts facts = query_facts();
    document["facts"] = facts_section(facts);
    warm_up_gpu();
    document["transfer"] = transfer_section();
    document["window"] = window_section();
    document["overlap"] = overlap_section();
    document["managed"] = managed_section(facts);
    document["in_vram"] = in_vram_section();
    std::printf("\n");
    write_json(document);
}

}  // namespace

// Report-only, like BM_CudaMemoryReport.
static void BM_CudaTieredMemoryReport(benchmark::State& state) {
    for (auto _ : state) {
        report();
        state.SkipWithError("report-only benchmark: see the printed tables");
        break;
    }
}
BENCHMARK(BM_CudaTieredMemoryReport)->Iterations(1)->Unit(benchmark::kSecond);
