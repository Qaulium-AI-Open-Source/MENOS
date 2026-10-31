// The Phase 22 step-1 CUDA memory benchmark (docs/adr/0026 sections 4 and 6).
//
// It measures, before any engine code changes, what the CUDA dense engine
// pays today for allocating, zero-filling, freeing and copying, against the
// CUDA runtime alternatives the ADR proposes -- all in the same run, because
// the GPU's clock state varies about 2x between runs (docs/adr/0020 section
// 10.4) and only same-run ratios are comparable:
//
//   (0) facts: runtime and driver versions, whether the device supports
//       stream-ordered memory pools and host registration, and the default
//       pool's release threshold;
//   (1) allocation: cudaMalloc + cudaMemset + cudaFree of a 16 * 2^n state,
//       as DeviceBuffer's constructor and destructor do it, against the same
//       allocation from a warm explicit memory pool (cudaMallocFromPoolAsync,
//       cudaFreeAsync, stream synchronize), and one cudaMemset on its own --
//       the second zero fill initialize() makes;
//   (2) the expectation scratch: what expectation_of_term pays per term
//       beyond its kernel (a 16 KiB cudaMalloc, a synchronising 16 KiB copy
//       back, a cudaFree and a host vector) against the copy alone into
//       buffers allocated once; and the whole call, kernel included, so the
//       overhead's share is visible;
//   (3) device-to-host: the 16 * 2^n copy into a pageable std::vector, as
//       DeviceBuffer::copy_to_host does it, against (a) cudaHostRegister of
//       the destination, copy, unregister, and (b) staging through two
//       page-locked chunks with cudaMemcpyAsync overlapping a host memcpy;
//   (4) end to end: CudaDenseEngine::execute on the crossover circuit family
//       (crossover_circuit.h) for a statevector, a one-term expectation and
//       a 100-term expectation, so (1)-(3) can be read as a share of a whole
//       execution (the materiality rule of docs/adr/0026 section 6, step 1).
//
// Every point is timed kRepetitions times after a warm-up, and the spread is
// reported beside the median (benchmark_spread.h). This is a benchmark, not a
// test: it asserts nothing. The destination vectors of (3) are allocated and
// touched once, outside the timing, so every path copies into resident pages
// and the comparison is of the transfer alone.
//
// Setting MENOS_CUDA_MEMORY_JSON=<path> also writes every table as JSON, for
// docs/benchmarks/phase-22-baseline/. As in cuda_crossover_benchmark.cpp it
// is an environment variable because the shared main() hands argv to Google
// Benchmark.
//
// This file includes <cuda_runtime.h> directly, as src/kernels/cuda/
// device_buffer.cpp does: the allocation and transfer calls it times are the
// runtime's own, and wrapping each in a probe would measure the wrapper.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
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
#include "menos/observable.h"
#include "menos/simulation_engine.h"
#include "menos_benchmark_environment.h"

namespace {

using Clock = std::chrono::steady_clock;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;
using menos::kernels::cuda::DeviceBuffer;
using menos::kernels::cuda::ensure_cuda;

// Repetitions per point, as in Phases 15 and 16 (owner, docs/adr/0019 Q1).
constexpr int kRepetitions = 10;

// Allocation cycles per sample in (1) and (2): a single cycle of a small
// buffer is microseconds, below the clock's useful resolution.
constexpr int kCyclesPerSample = 10;

// Observable terms in (2) and in the many-term row of (4): the count the ADR
// names (docs/adr/0026 section 1.3).
constexpr int kTerms = 100;

constexpr std::uint32_t kMinAllocationQubits = 16;
constexpr std::uint32_t kMinTransferQubits = 20;
constexpr std::uint32_t kMaxQubits = 26;

constexpr double kMicro = 1e6;
constexpr double kGiga = 1e9;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

nlohmann::json spread_json(const Spread& s) {
    return {{"median", s.median}, {"min", s.min}, {"max", s.max}, {"q1", s.q1}, {"q3", s.q3}};
}

std::uint64_t state_bytes(std::uint32_t num_qubits) {
    return std::uint64_t{16} << num_qubits;
}

// kRepetitions samples, each the mean host wall time over `cycles` calls of
// `call`, after one untimed call.
Spread time_host(const std::function<void()>& call, int cycles) {
    call();
    std::vector<double> samples;
    for (int r = 0; r < kRepetitions; ++r) {
        const Clock::time_point start = Clock::now();
        for (int c = 0; c < cycles; ++c) {
            call();
        }
        samples.push_back(seconds_since(start) / cycles);
    }
    return spread(std::move(samples));
}

// ---------------------------------------------------------------------------
// (0) facts
// ---------------------------------------------------------------------------

struct Facts {
    int runtime_version = 0;
    int driver_version = 0;
    int memory_pools_supported = 0;
    int host_register_supported = 0;
    std::uint64_t default_pool_release_threshold = 0;
};

Facts query_facts() {
    Facts facts;
    ensure_cuda(cudaRuntimeGetVersion(&facts.runtime_version), "cudaRuntimeGetVersion");
    ensure_cuda(cudaDriverGetVersion(&facts.driver_version), "cudaDriverGetVersion");
    ensure_cuda(
        cudaDeviceGetAttribute(&facts.memory_pools_supported, cudaDevAttrMemoryPoolsSupported, 0),
        "cudaDeviceGetAttribute (MemoryPoolsSupported)");
    ensure_cuda(
        cudaDeviceGetAttribute(&facts.host_register_supported, cudaDevAttrHostRegisterSupported, 0),
        "cudaDeviceGetAttribute (HostRegisterSupported)");
    if (facts.memory_pools_supported != 0) {
        cudaMemPool_t pool = nullptr;
        ensure_cuda(cudaDeviceGetDefaultMemPool(&pool, 0), "cudaDeviceGetDefaultMemPool");
        std::uint64_t threshold = 0;
        ensure_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold),
                    "cudaMemPoolGetAttribute (ReleaseThreshold)");
        facts.default_pool_release_threshold = threshold;
    }
    return facts;
}

nlohmann::json facts_section(const Facts& facts) {
    std::printf("\n== (0) facts ==\n");
    std::printf("runtime version          : %d\n", facts.runtime_version);
    std::printf("driver version           : %d\n", facts.driver_version);
    std::printf("memory pools supported   : %d\n", facts.memory_pools_supported);
    std::printf("host register supported  : %d\n", facts.host_register_supported);
    std::printf("default pool threshold   : %llu bytes\n",
                static_cast<unsigned long long>(facts.default_pool_release_threshold));
    return {
        {"runtime_version", facts.runtime_version},
        {"driver_version", facts.driver_version},
        {"memory_pools_supported", facts.memory_pools_supported},
        {"host_register_supported", facts.host_register_supported},
        {"default_pool_release_threshold_bytes", facts.default_pool_release_threshold},
    };
}

// ---------------------------------------------------------------------------
// (1) allocation
// ---------------------------------------------------------------------------

// An explicit stream-ordered pool with its release threshold at the device's
// total memory, the configuration docs/adr/0026 section 3.1 proposes (Q2), so
// the memory a cycle frees stays in the pool for the next cycle.
class ExplicitPool {
public:
    explicit ExplicitPool(std::uint64_t release_threshold) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = 0;
        ensure_cuda(cudaMemPoolCreate(&pool_, &props), "cudaMemPoolCreate");
        std::uint64_t threshold = release_threshold;
        ensure_cuda(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &threshold),
                    "cudaMemPoolSetAttribute (ReleaseThreshold)");
    }
    ~ExplicitPool() {
        static_cast<void>(cudaMemPoolDestroy(pool_));
    }
    ExplicitPool(const ExplicitPool&) = delete;
    ExplicitPool& operator=(const ExplicitPool&) = delete;
    ExplicitPool(ExplicitPool&&) = delete;
    ExplicitPool& operator=(ExplicitPool&&) = delete;

    // Allocate, free and wait, on the legacy default stream the engine uses.
    void cycle(std::size_t bytes) const {
        void* data = nullptr;
        ensure_cuda(cudaMallocFromPoolAsync(&data, bytes, pool_, nullptr),
                    "cudaMallocFromPoolAsync");
        ensure_cuda(cudaFreeAsync(data, nullptr), "cudaFreeAsync");
        ensure_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
    }

    [[nodiscard]] cudaMemPool_t handle() const {
        return pool_;
    }

    [[nodiscard]] std::uint64_t reserved_bytes() const {
        std::uint64_t value = 0;
        ensure_cuda(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &value),
                    "cudaMemPoolGetAttribute (ReservedMemCurrent)");
        return value;
    }

private:
    cudaMemPool_t pool_ = nullptr;
};

// What DeviceBuffer's constructor and destructor do today
// (src/kernels/cuda/device_buffer.cpp): allocate, zero-fill, free.
void direct_cycle(std::size_t bytes) {
    void* data = nullptr;
    ensure_cuda(cudaMalloc(&data, bytes), "cudaMalloc");
    ensure_cuda(cudaMemset(data, 0, bytes), "cudaMemset");
    ensure_cuda(cudaFree(data), "cudaFree");
    ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

struct AllocationRow {
    Spread direct;
    Spread memset_only;
    Spread pooled;
};

nlohmann::json allocation_section(const Facts& facts,
                                  std::map<std::uint32_t, AllocationRow>& rows_out) {
    std::printf(
        "\n== (1) allocation: cudaMalloc + cudaMemset + cudaFree against a warm pool "
        "(microseconds, medians) ==\n");
    std::printf("%3s %11s %13s %13s %13s %9s %13s\n",
                "n",
                "bytes",
                "direct",
                "memset only",
                "pooled",
                "d/pooled",
                "saved/cycle");
    nlohmann::json rows = nlohmann::json::array();
    const menos::kernels::cuda::DeviceInfo& info = menos::kernels::cuda::device_info();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    std::optional<ExplicitPool> pool;
    if (facts.memory_pools_supported != 0) {
        pool.emplace(info.total_memory_bytes);
    }
    for (std::uint32_t n = kMinAllocationQubits; n <= kMaxQubits; n += 2) {
        const std::uint64_t bytes = state_bytes(n);
        if (bytes > device_budget) {
            std::printf("%3u (skipped: does not fit the device)\n", n);
            rows.push_back({{"n", n}, {"measured", false}});
            continue;
        }
        const auto size = static_cast<std::size_t>(bytes);
        const Spread direct = time_host([size] { direct_cycle(size); }, kCyclesPerSample);
        Spread memset_only{};
        {
            DeviceBuffer held(size / sizeof(std::complex<double>), device_budget);
            void* data = held.state().data;
            memset_only = time_host(
                [data, size] {
                    ensure_cuda(cudaMemset(data, 0, size), "cudaMemset");
                    ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                },
                kCyclesPerSample);
        }
        Spread pooled{};
        if (pool.has_value()) {
            pooled = time_host([&pool, size] { pool->cycle(size); }, kCyclesPerSample);
        }
        const double ratio = pool.has_value() ? direct.median / pooled.median : 0.0;
        const double saved = pool.has_value() ? direct.median - pooled.median : 0.0;
        std::printf("%3u %11llu %13.2f %13.2f %13.2f %8.2fx %13.2f\n",
                    n,
                    static_cast<unsigned long long>(bytes),
                    direct.median * kMicro,
                    memset_only.median * kMicro,
                    pooled.median * kMicro,
                    ratio,
                    saved * kMicro);
        std::fflush(stdout);
        rows_out[n] = AllocationRow{direct, memset_only, pooled};
        rows.push_back({
            {"n", n},
            {"measured", true},
            {"bytes", bytes},
            {"direct_cycle_seconds", spread_json(direct)},
            {"memset_only_seconds", spread_json(memset_only)},
            {"pooled_cycle_seconds", spread_json(pooled)},
            {"pooled_measured", pool.has_value()},
            {"direct_over_pooled", ratio},
            {"saved_per_cycle_seconds", saved},
        });
    }
    if (pool.has_value()) {
        std::printf("pool reserved after the sweep: %llu bytes\n",
                    static_cast<unsigned long long>(pool->reserved_bytes()));
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (2) the expectation scratch
// ---------------------------------------------------------------------------

// An observable of `count` distinct single- and two-qubit Pauli terms on n
// qubits: Z on qubit t mod n, and on odd terms also X on another qubit, so
// no two consecutive terms are alike. The values are not the point; the
// per-term costs are.
// num_qubits (a width) and count (how many terms) are not confusable at the
// two call sites, which pass the named constants.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
menos::Observable many_terms(std::uint32_t num_qubits, int count) {
    menos::Observable observable;
    for (int t = 0; t < count; ++t) {
        menos::PauliString pauli;
        pauli.num_qubits = num_qubits;
        const auto z_qubit = static_cast<std::uint32_t>(t) % num_qubits;
        const auto x_qubit = (static_cast<std::uint32_t>(t) * 7U + 3U) % num_qubits;
        pauli.z_words = {std::uint64_t{1} << z_qubit};
        pauli.x_words = {(t % 2 == 1) ? (std::uint64_t{1} << x_qubit) : std::uint64_t{0}};
        observable.terms.emplace_back(pauli, std::complex<double>{1.0, 0.0});
    }
    return observable;
}

nlohmann::json scratch_section(const Facts& facts) {
    constexpr std::uint32_t kScratchQubits = 20;
    const std::size_t scratch_bytes = menos::kernels::cuda::expectation_scratch_bytes();
    std::printf(
        "\n== (2) the expectation scratch per term at n = %u (microseconds, medians over "
        "%d terms) ==\n",
        kScratchQubits,
        kTerms);
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    if (state_bytes(kScratchQubits) + scratch_bytes > device_budget) {
        std::printf("(skipped: does not fit the device)\n");
        return {{"measured", false}};
    }
    DeviceBuffer state(std::size_t{1} << kScratchQubits, device_budget);
    menos::kernels::cuda::initialize(state.state(), 0.0);

    // Today's per-term bookkeeping, exactly as dense_kernels.cu does it minus
    // the kernel: allocate the partials, copy them back, free them, and a
    // fresh host vector to receive them.
    const auto per_term_today = [scratch_bytes] {
        void* partials = nullptr;
        ensure_cuda(cudaMalloc(&partials, scratch_bytes), "cudaMalloc (partials)");
        std::vector<std::complex<double>> host(menos::kernels::cuda::kExpectationMaxBlocks);
        ensure_cuda(cudaMemcpy(host.data(), partials, scratch_bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy (partials)");
        ensure_cuda(cudaFree(partials), "cudaFree (partials)");
        benchmark::DoNotOptimize(host.data());
    };
    // The proposal: the same copy into buffers allocated once per execution.
    DeviceBuffer scratch(menos::kernels::cuda::kExpectationMaxBlocks, device_budget);
    std::vector<std::complex<double>> reused_host(menos::kernels::cuda::kExpectationMaxBlocks);
    const auto per_term_reused = [&scratch, &reused_host, scratch_bytes] {
        ensure_cuda(
            cudaMemcpy(
                reused_host.data(), scratch.state().data, scratch_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy (reused partials)");
        benchmark::DoNotOptimize(reused_host.data());
    };
    // The whole call as shipped, kernel included, on the many-term observable.
    const menos::Observable observable = many_terms(kScratchQubits, kTerms);
    std::size_t next = 0;
    const auto whole_call = [&state, &observable, &next] {
        const auto& term = observable.terms[next % observable.terms.size()].first;
        ++next;
        benchmark::DoNotOptimize(menos::kernels::cuda::expectation_of_term(
            state.state(), term.x_words[0], term.z_words[0]));
    };

    // Today's bookkeeping with the allocation served from a warm pool instead
    // of cudaMalloc: what item 1 alone would do to a term.
    std::optional<ExplicitPool> pool;
    if (facts.memory_pools_supported != 0) {
        pool.emplace(menos::kernels::cuda::device_info().total_memory_bytes);
    }
    const auto per_term_pooled = [&pool, scratch_bytes] {
        void* partials = nullptr;
        ensure_cuda(cudaMallocFromPoolAsync(&partials, scratch_bytes, pool->handle(), nullptr),
                    "cudaMallocFromPoolAsync (partials)");
        std::vector<std::complex<double>> host(menos::kernels::cuda::kExpectationMaxBlocks);
        ensure_cuda(cudaMemcpy(host.data(), partials, scratch_bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy (pooled partials)");
        ensure_cuda(cudaFreeAsync(partials, nullptr), "cudaFreeAsync (partials)");
        ensure_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize (partials)");
        benchmark::DoNotOptimize(host.data());
    };

    const Spread today = time_host(per_term_today, kTerms);
    const Spread reused = time_host(per_term_reused, kTerms);
    const Spread pooled = pool.has_value() ? time_host(per_term_pooled, kTerms) : Spread{};
    const Spread whole = time_host(whole_call, kTerms);

    // The same call on the state the crossover circuit leaves (every
    // amplitude non-zero), and with one gate between terms, as the engine
    // runs it after a gate sweep: the first draft of this benchmark found the
    // engine paying far more per term than the loop above, and these two
    // rows say whether the state's contents or the preceding gate is why.
    const menos::Circuit circuit = menos::benchmarks::make_crossover_circuit(kScratchQubits);
    for (const menos::Operation& operation : circuit.operations()) {
        if (const auto* gate = std::get_if<menos::GateOp>(&operation)) {
            menos::kernels::cuda::apply_matrix(
                state.state(),
                menos::ir::generate_matrix(gate->opcode, gate->params),
                gate->qubits);
        }
    }
    const Spread whole_dense = time_host(whole_call, kTerms);
    const std::vector<std::complex<double>> hadamard =
        menos::ir::generate_matrix(menos::Opcode::H, {});
    const std::vector<std::uint32_t> qubit_zero{0};
    const auto gate_only = [&state, &hadamard, &qubit_zero] {
        menos::kernels::cuda::apply_matrix(state.state(), hadamard, qubit_zero);
    };
    const auto interleaved = [&gate_only, &whole_call] {
        gate_only();
        whole_call();
    };
    const Spread gate = time_host(gate_only, kTerms);
    const Spread gate_and_term = time_host(interleaved, kTerms);
    const double after_gate = gate_and_term.median - gate.median;

    // Exactly execute()'s own sequence: a buffer allocated for this run,
    // initialize, the gate sweep, then the kTerms terms, with the term loop
    // timed on its own. If this disagrees with the rows above, the engine's
    // per-term cost depends on the buffer being fresh, not on the call.
    std::vector<double> fresh_samples;
    for (int r = -1; r < kRepetitions; ++r) {
        DeviceBuffer fresh(std::size_t{1} << kScratchQubits, device_budget);
        menos::kernels::cuda::initialize(fresh.state(), 0.0);
        for (const menos::Operation& operation : circuit.operations()) {
            if (const auto* gate_op = std::get_if<menos::GateOp>(&operation)) {
                menos::kernels::cuda::apply_matrix(
                    fresh.state(),
                    menos::ir::generate_matrix(gate_op->opcode, gate_op->params),
                    gate_op->qubits);
            }
        }
        const Clock::time_point start = Clock::now();
        for (const auto& term : observable.terms) {
            benchmark::DoNotOptimize(menos::kernels::cuda::expectation_of_term(
                fresh.state(), term.first.x_words[0], term.first.z_words[0]));
        }
        if (r >= 0) {
            fresh_samples.push_back(seconds_since(start) / kTerms);
        }
    }
    const Spread in_sequence = spread(std::move(fresh_samples));

    const double share = (today.median - reused.median) / whole.median;
    std::printf("%-36s %13.2f\n", "bookkeeping today", today.median * kMicro);
    std::printf("%-36s %13.2f\n", "bookkeeping, allocation pooled", pooled.median * kMicro);
    std::printf("%-36s %13.2f\n", "copy into reused scratch", reused.median * kMicro);
    std::printf("%-36s %13.2f\n", "whole expectation_of_term, |0>", whole.median * kMicro);
    std::printf("%-36s %13.2f\n", "whole expectation_of_term, dense", whole_dense.median * kMicro);
    std::printf("%-36s %13.2f\n", "one H gate", gate.median * kMicro);
    std::printf("%-36s %13.2f\n", "H gate then term, minus the gate", after_gate * kMicro);
    std::printf("%-36s %13.2f\n", "term in execute()'s own sequence", in_sequence.median * kMicro);
    std::printf("%-36s %12.1f%%\n", "removable share of a |0> term", share * 100.0);
    std::fflush(stdout);
    return {
        {"measured", true},
        {"n", kScratchQubits},
        {"terms", kTerms},
        {"scratch_bytes", scratch_bytes},
        {"bookkeeping_today_seconds", spread_json(today)},
        {"bookkeeping_pooled_seconds", spread_json(pooled)},
        {"pooled_measured", pool.has_value()},
        {"reused_copy_seconds", spread_json(reused)},
        {"whole_call_seconds", spread_json(whole)},
        {"whole_call_dense_state_seconds", spread_json(whole_dense)},
        {"gate_only_seconds", spread_json(gate)},
        {"gate_then_term_seconds", spread_json(gate_and_term)},
        {"term_after_gate_seconds", after_gate},
        {"term_in_execute_sequence_seconds", spread_json(in_sequence)},
        {"removable_share", share},
    };
}

// ---------------------------------------------------------------------------
// (3) device-to-host
// ---------------------------------------------------------------------------

// Two page-locked chunks and a stream of their own: the copy of chunk k + 1
// is in flight while chunk k is drained into the destination on the host.
class StagedCopier {
public:
    explicit StagedCopier(std::size_t chunk_bytes) : chunk_bytes_(chunk_bytes) {
        for (std::size_t i = 0; i < 2; ++i) {
            ensure_cuda(cudaHostAlloc(&chunks_[i], chunk_bytes, cudaHostAllocDefault),
                        "cudaHostAlloc");
            ensure_cuda(cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming),
                        "cudaEventCreateWithFlags");
        }
        ensure_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags");
    }
    ~StagedCopier() {
        static_cast<void>(cudaStreamDestroy(stream_));
        for (std::size_t i = 0; i < 2; ++i) {
            static_cast<void>(cudaEventDestroy(events_[i]));
            static_cast<void>(cudaFreeHost(chunks_[i]));
        }
    }
    StagedCopier(const StagedCopier&) = delete;
    StagedCopier& operator=(const StagedCopier&) = delete;
    StagedCopier(StagedCopier&&) = delete;
    StagedCopier& operator=(StagedCopier&&) = delete;

    void copy(std::span<std::complex<double>> destination, const void* source) const {
        auto* dest = reinterpret_cast<unsigned char*>(destination.data());
        const auto* src = static_cast<const unsigned char*>(source);
        const std::size_t total = destination.size_bytes();
        const std::size_t count = (total + chunk_bytes_ - 1) / chunk_bytes_;
        const auto issue = [&](std::size_t k) {
            const std::size_t offset = k * chunk_bytes_;
            const std::size_t size = std::min(chunk_bytes_, total - offset);
            ensure_cuda(cudaMemcpyAsync(
                            chunks_[k % 2], src + offset, size, cudaMemcpyDeviceToHost, stream_),
                        "cudaMemcpyAsync (staged)");
            ensure_cuda(cudaEventRecord(events_[k % 2], stream_), "cudaEventRecord");
        };
        issue(0);
        for (std::size_t k = 0; k < count; ++k) {
            ensure_cuda(cudaEventSynchronize(events_[k % 2]), "cudaEventSynchronize");
            if (k + 1 < count) {
                issue(k + 1);
            }
            const std::size_t offset = k * chunk_bytes_;
            const std::size_t size = std::min(chunk_bytes_, total - offset);
            std::memcpy(dest + offset, chunks_[k % 2], size);
        }
    }

private:
    std::size_t chunk_bytes_;
    void* chunks_[2] = {nullptr, nullptr};
    cudaEvent_t events_[2] = {nullptr, nullptr};
    cudaStream_t stream_ = nullptr;
};

nlohmann::json transfer_section(const Facts& facts, std::map<std::uint32_t, Spread>& pageable_out) {
    std::printf(
        "\n== (3) device-to-host of the state (milliseconds, medians; GB/s of the "
        "median) ==\n");
    std::printf("%3s %10s %7s %10s %10s %10s %7s %10s %7s %10s %7s %10s %7s\n",
                "n",
                "pageable",
                "GB/s",
                "register",
                "reg copy",
                "reg total",
                "GB/s",
                "staged 4M",
                "GB/s",
                "staged16M",
                "GB/s",
                "staged64M",
                "GB/s");
    nlohmann::json rows = nlohmann::json::array();
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    constexpr std::size_t kChunkSizes[] = {
        std::size_t{4} << 20, std::size_t{16} << 20, std::size_t{64} << 20};
    for (std::uint32_t n = kMinTransferQubits; n <= kMaxQubits; n += 2) {
        const std::uint64_t bytes = state_bytes(n);
        if (bytes > device_budget || bytes + 2 * kChunkSizes[2] > host_budget) {
            std::printf("%3u (skipped: does not fit the device or host budget)\n", n);
            rows.push_back({{"n", n}, {"measured", false}});
            continue;
        }
        const std::size_t amplitudes = std::size_t{1} << n;
        DeviceBuffer source(amplitudes, device_budget);
        menos::kernels::cuda::initialize(source.state(), 0.0);
        // Resident before any timing (see the file comment).
        std::vector<std::complex<double>> destination(amplitudes);
        std::fill(destination.begin(), destination.end(), std::complex<double>{1.0, 1.0});
        const double gb = static_cast<double>(bytes) / kGiga;

        const Spread pageable = time_host([&] { source.copy_to_host(destination); }, 1);
        pageable_out[n] = pageable;

        Spread registration{};
        Spread registered_copy{};
        Spread unregistration{};
        if (facts.host_register_supported != 0) {
            // The three parts timed separately, each kRepetitions times, so
            // the fixed cost of locking the pages is visible on its own.
            std::vector<double> reg_samples;
            std::vector<double> copy_samples;
            std::vector<double> unreg_samples;
            for (int r = -1; r < kRepetitions; ++r) {
                const Clock::time_point t0 = Clock::now();
                ensure_cuda(cudaHostRegister(destination.data(),
                                             destination.size() * sizeof(std::complex<double>),
                                             cudaHostRegisterDefault),
                            "cudaHostRegister");
                const Clock::time_point t1 = Clock::now();
                source.copy_to_host(destination);
                const Clock::time_point t2 = Clock::now();
                ensure_cuda(cudaHostUnregister(destination.data()), "cudaHostUnregister");
                const Clock::time_point t3 = Clock::now();
                if (r >= 0) {
                    reg_samples.push_back(std::chrono::duration<double>(t1 - t0).count());
                    copy_samples.push_back(std::chrono::duration<double>(t2 - t1).count());
                    unreg_samples.push_back(std::chrono::duration<double>(t3 - t2).count());
                }
            }
            registration = spread(std::move(reg_samples));
            registered_copy = spread(std::move(copy_samples));
            unregistration = spread(std::move(unreg_samples));
        }
        const double registered_total =
            registration.median + registered_copy.median + unregistration.median;

        std::vector<Spread> staged;
        for (const std::size_t chunk : kChunkSizes) {
            const StagedCopier copier(chunk);
            staged.push_back(time_host(
                [&copier, &destination, &source] { copier.copy(destination, source.state().data); },
                1));
        }

        constexpr double kMilli = 1e3;
        std::printf(
            "%3u %10.3f %7.2f %10.3f %10.3f %10.3f %7.2f %10.3f %7.2f %10.3f %7.2f %10.3f "
            "%7.2f\n",
            n,
            pageable.median * kMilli,
            gb / pageable.median,
            registration.median * kMilli,
            registered_copy.median * kMilli,
            registered_total * kMilli,
            registered_total > 0.0 ? gb / registered_total : 0.0,
            staged[0].median * kMilli,
            gb / staged[0].median,
            staged[1].median * kMilli,
            gb / staged[1].median,
            staged[2].median * kMilli,
            gb / staged[2].median);
        std::fflush(stdout);
        rows.push_back({
            {"n", n},
            {"measured", true},
            {"bytes", bytes},
            {"pageable_seconds", spread_json(pageable)},
            {"register_seconds", spread_json(registration)},
            {"registered_copy_seconds", spread_json(registered_copy)},
            {"unregister_seconds", spread_json(unregistration)},
            {"registered_total_seconds", registered_total},
            {"staged_4mib_seconds", spread_json(staged[0])},
            {"staged_16mib_seconds", spread_json(staged[1])},
            {"staged_64mib_seconds", spread_json(staged[2])},
        });
    }
    return rows;
}

// ---------------------------------------------------------------------------
// (4) end to end
// ---------------------------------------------------------------------------

Spread time_execute(const menos::engines::dense_cuda::CudaDenseEngine& engine,
                    const menos::EngineExecutionRequest& request) {
    std::vector<double> samples;
    for (int r = -1; r < kRepetitions; ++r) {
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

nlohmann::json end_to_end_section(const std::map<std::uint32_t, AllocationRow>& allocation,
                                  const std::map<std::uint32_t, Spread>& pageable) {
    std::printf(
        "\n== (4) CudaDenseEngine::execute end to end (seconds, medians) and the share "
        "(1)-(3) could remove ==\n");
    std::printf("%3s %5s %11s %11s %11s %9s %9s %9s %9s %9s\n",
                "n",
                "gates",
                "statevec",
                "exp 1 term",
                "exp 100",
                "alloc/sv",
                "memset/sv",
                "d2h/sv",
                "alloc/e1",
                "per term");
    nlohmann::json rows = nlohmann::json::array();
    const menos::engines::dense_cuda::CudaDenseEngine engine;
    const std::uint64_t device_budget = menos::kernels::cuda::device_available_bytes();
    const std::uint64_t host_budget = menos_benchmark::safe_allocation_budget_bytes();
    for (std::uint32_t n = kMinAllocationQubits; n <= kMaxQubits; n += 2) {
        const std::uint64_t bytes = state_bytes(n);
        if (bytes + menos::kernels::cuda::expectation_scratch_bytes() > device_budget ||
            bytes > host_budget) {
            std::printf("%3u (skipped: does not fit the device or host budget)\n", n);
            rows.push_back({{"n", n}, {"measured", false}});
            continue;
        }
        const menos::Circuit circuit = menos::benchmarks::make_crossover_circuit(n);
        const menos::Observable one = many_terms(n, 1);
        const menos::Observable many = many_terms(n, kTerms);
        const auto request = [&](bool want_statevector, const menos::Observable* observable) {
            return menos::EngineExecutionRequest{
                .num_qubits = n,
                .operations = circuit.operations(),
                .want_statevector = want_statevector,
                .want_counts = false,
                .want_expectation = observable != nullptr,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = 0,
                .global_phase = 0.0,
                .budget_bytes = host_budget,
                .is_sole_region = true,
                .observable = observable,
            };
        };
        const Spread statevector = time_execute(engine, request(true, nullptr));
        const Spread one_term = time_execute(engine, request(false, &one));
        const Spread many_term = time_execute(engine, request(false, &many));

        const auto alloc = allocation.find(n);
        const auto d2h = pageable.find(n);
        const double alloc_saving = alloc == allocation.end()
                                        ? 0.0
                                        : alloc->second.direct.median - alloc->second.pooled.median;
        const double memset_cost =
            alloc == allocation.end() ? 0.0 : alloc->second.memset_only.median;
        const double d2h_cost = d2h == pageable.end() ? 0.0 : d2h->second.median;
        const double per_extra_term = (many_term.median - one_term.median) / (kTerms - 1);
        std::printf("%3u %5zu %11.6f %11.6f %11.6f %8.2f%% %8.2f%% %8.2f%% %8.2f%% %9.1f\n",
                    n,
                    circuit.operations().size(),
                    statevector.median,
                    one_term.median,
                    many_term.median,
                    alloc_saving / statevector.median * 100.0,
                    memset_cost / statevector.median * 100.0,
                    d2h_cost / statevector.median * 100.0,
                    alloc_saving / one_term.median * 100.0,
                    per_extra_term * kMicro);
        std::fflush(stdout);
        rows.push_back({
            {"n", n},
            {"measured", true},
            {"gates", circuit.operations().size()},
            {"statevector_seconds", spread_json(statevector)},
            {"expectation_1_term_seconds", spread_json(one_term)},
            {"expectation_100_terms_seconds", spread_json(many_term)},
            {"allocation_saving_share_of_statevector", alloc_saving / statevector.median},
            {"memset_share_of_statevector", memset_cost / statevector.median},
            {"pageable_d2h_share_of_statevector", d2h_cost / statevector.median},
            {"allocation_saving_share_of_expectation_1", alloc_saving / one_term.median},
            {"per_extra_term_seconds", per_extra_term},
        });
    }
    return rows;
}

// About a second of full-bandwidth work before anything is timed, so the first
// rows are not measured while the GPU is still raising its clocks from idle
// (cuda_kernel_benchmark.cpp learned this the hard way).
void warm_up_gpu() {
    constexpr std::uint32_t kWarmQubits = 24;
    constexpr int kWarmPasses = 100;
    DeviceBuffer buffer(std::size_t{1} << kWarmQubits,
                        menos::kernels::cuda::device_available_bytes());
    const std::size_t bytes = state_bytes(kWarmQubits);
    for (int pass = 0; pass < kWarmPasses; ++pass) {
        ensure_cuda(cudaMemset(buffer.state().data, pass & 1, bytes), "cudaMemset (warm-up)");
    }
    ensure_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warm-up)");
}

void write_json(const nlohmann::json& document) {
    const char* const path = std::getenv("MENOS_CUDA_MEMORY_JSON");
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
    std::printf("\n=== MENOS CUDA memory benchmark (Phase 22) ===\n");
    nlohmann::json document = {{"schema_version", 1}, {"device_present", info.present}};
    if (!info.present) {
        std::printf("no CUDA device available -- nothing measured\n");
        write_json(document);
        return;
    }
    std::printf("environment      : %s\n", menos_benchmark::describe_environment().c_str());
    std::printf("GPU              : %s (compute capability %d.%d, %d SMs, %llu bytes total)\n",
                info.name.c_str(),
                info.compute_capability_major,
                info.compute_capability_minor,
                info.multiprocessor_count,
                static_cast<unsigned long long>(info.total_memory_bytes));
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
    document["cycles_per_sample"] = kCyclesPerSample;
    document["terms"] = kTerms;
    const Facts facts = query_facts();
    document["facts"] = facts_section(facts);
    warm_up_gpu();
    std::map<std::uint32_t, AllocationRow> allocation_rows;
    std::map<std::uint32_t, Spread> pageable_rows;
    document["allocation"] = allocation_section(facts, allocation_rows);
    document["scratch"] = scratch_section(facts);
    document["transfer"] = transfer_section(facts, pageable_rows);
    document["end_to_end"] = end_to_end_section(allocation_rows, pageable_rows);
    std::printf("\n");
    write_json(document);
}

}  // namespace

// Report-only, like BM_CudaKernelReport.
static void BM_CudaMemoryReport(benchmark::State& state) {
    for (auto _ : state) {
        report();
        state.SkipWithError("report-only benchmark: see the printed tables");
        break;
    }
}
BENCHMARK(BM_CudaMemoryReport)->Iterations(1)->Unit(benchmark::kSecond);
