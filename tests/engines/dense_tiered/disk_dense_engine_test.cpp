// Phase 23 (docs/adr/0027 sections 3.2 and 3.4): the disk tier's outputs are
// BITWISE the in-memory DenseEngine's for the same request. Random circuits
// at n = 8-14 run through the internal chunk and window seams at every chunk
// size from a handful of amplitudes to the whole state, with gates on local,
// global and straddling operands of one to five qubits, on every thread
// count; the counts and expectation terms are compared byte for byte.

#include "engines/dense_tiered/disk_dense_engine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/memory_ledger.h"
#include "common/thread_count.h"
#include "engines/dense/dense_engine.h"
#include "engines/dense_tiered/chunked_executor.h"
#include "ir/fuse_gates.h"
#include "menos/circuit.h"
#include "menos/observable.h"

namespace menos::engines::dense_tiered {
namespace {

using Complex = std::complex<double>;

// Draws straight from the engine's standardised output, not through the
// standard distributions, whose sequences differ between standard libraries:
// the circuits below must be the same on every platform, so the chunk and
// window seams a test chooses fit their gates everywhere.
std::uint64_t below(std::mt19937_64& rng, std::uint64_t bound) {
    return rng() % bound;
}

double angle_of(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * (6.283185307179586 / 9007199254740992.0);
}

// A random single-qubit unitary, exactly unitary by construction.
std::vector<Complex> random_1q_unitary(std::mt19937_64& rng) {
    const double a = angle_of(rng);
    const double b = angle_of(rng);
    const double c = angle_of(rng);
    const Complex eb = std::polar(1.0, b);
    const Complex ec = std::polar(1.0, c);
    return {Complex{std::cos(a), 0.0}, -eb * std::sin(a), ec * std::sin(a), eb * ec * std::cos(a)};
}

std::vector<Complex> kron(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    // Operand order: bit j of the row/column index is operand j, so the
    // product's lower-index factor is the lower bit.
    const std::size_t da = static_cast<std::size_t>(std::sqrt(static_cast<double>(a.size())));
    const std::size_t db = static_cast<std::size_t>(std::sqrt(static_cast<double>(b.size())));
    const std::size_t d = da * db;
    std::vector<Complex> m(d * d);
    for (std::size_t r = 0; r < d; ++r) {
        for (std::size_t c = 0; c < d; ++c) {
            m[r * d + c] = a[(r % da) * da + (c % da)] * b[(r / da) * db + (c / da)];
        }
    }
    return m;
}

// k distinct qubits of n, a Fisher-Yates prefix.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<QubitId> distinct_qubits(std::uint32_t n, std::size_t k, std::mt19937_64& rng) {
    std::vector<QubitId> all(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        all[q] = q;
    }
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t j = i + static_cast<std::size_t>(below(rng, n - i));
        std::swap(all[i], all[j]);
    }
    all.resize(k);
    return all;
}

// A random circuit over every gate shape the executor remaps: one-, two-
// and three-operand named gates, and two- and five-operand unitaries.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Circuit random_circuit(std::uint32_t n, int gates, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto angle = [&rng]() { return angle_of(rng); };
    Circuit circuit(n, n);
    for (int g = 0; g < gates; ++g) {
        switch (below(rng, 12)) {
            case 0:
                circuit.append(Opcode::H, distinct_qubits(n, 1, rng));
                break;
            case 1:
                circuit.append(Opcode::RX, distinct_qubits(n, 1, rng), {angle()});
                break;
            case 2:
                circuit.append(Opcode::RZ, distinct_qubits(n, 1, rng), {angle()});
                break;
            case 3:
                circuit.append(Opcode::T, distinct_qubits(n, 1, rng));
                break;
            case 4:
            case 5:
                circuit.append(Opcode::CX, distinct_qubits(n, 2, rng));
                break;
            case 6:
                circuit.append(Opcode::CP, distinct_qubits(n, 2, rng), {angle()});
                break;
            case 7:
                circuit.append(Opcode::SWAP, distinct_qubits(n, 2, rng));
                break;
            case 8:
                circuit.append(Opcode::CCX, distinct_qubits(n, 3, rng));
                break;
            case 9:
                circuit.append_unitary(distinct_qubits(n, 2, rng),
                                       kron(random_1q_unitary(rng), random_1q_unitary(rng)));
                break;
            case 10: {
                std::vector<Complex> m = random_1q_unitary(rng);
                for (int i = 1; i < 5; ++i) {
                    m = kron(m, random_1q_unitary(rng));
                }
                circuit.append_unitary(distinct_qubits(n, 5, rng), m);
                break;
            }
            default:
                circuit.append(Opcode::CZ, distinct_qubits(n, 2, rng));
                break;
        }
    }
    for (QubitId q = 0; q < n; ++q) {
        circuit.append_measure(q, q);
    }
    return circuit;
}

std::vector<std::pair<QubitId, ClbitId>> all_targets(std::uint32_t n) {
    std::vector<std::pair<QubitId, ClbitId>> targets;
    for (QubitId q = 0; q < n; ++q) {
        targets.emplace_back(q, q);
    }
    return targets;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Observable random_observable(std::uint32_t n, int terms, std::mt19937_64& rng) {
    const std::uint64_t all = (std::uint64_t{1} << n) - 1;
    const auto pick = [&rng, all]() { return below(rng, all + 1); };
    Observable observable;
    observable.terms.reserve(3 + static_cast<std::size_t>(terms));
    // Z-only, X-only, a single high X bit (pairs every chunk with another),
    // and random Y-bearing terms.
    observable.terms.emplace_back(PauliString{n, {0}, {all}}, Complex{1.0, 0.0});
    observable.terms.emplace_back(PauliString{n, {all}, {0}}, Complex{0.5, 0.0});
    observable.terms.emplace_back(
        PauliString{n, {std::uint64_t{1} << (n - 1)}, {std::uint64_t{1} << (n - 1)}},
        Complex{-0.25, 0.0});
    for (int i = 0; i < terms; ++i) {
        observable.terms.emplace_back(PauliString{n, {pick()}, {pick()}}, Complex{0.1, 0.0});
    }
    return observable;
}

struct Outputs {
    std::map<BitString, std::uint64_t> counts;
    std::vector<Complex> terms;
};

Outputs run_in_memory(const Circuit& circuit,
                      std::uint64_t shots,
                      const Observable* observable,
                      double global_phase,
                      std::uint32_t threads) {
    const std::vector<std::pair<QubitId, ClbitId>> targets = all_targets(circuit.num_qubits());
    const EngineExecutionRequest request{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = false,
        .want_counts = shots > 0,
        .want_expectation = observable != nullptr,
        .shots = shots,
        .measure_targets = targets,
        .bitstring_width = circuit.num_clbits(),
        .seed = 7,
        .global_phase = global_phase,
        .budget_bytes = std::numeric_limits<std::uint64_t>::max(),
        .is_sole_region = true,
        .observable = observable,
        .max_threads = threads,
    };
    const dense::DenseEngine engine;
    EngineExecutionResult result = engine.execute(request);
    return Outputs{result.counts.value_or(std::map<BitString, std::uint64_t>{}),
                   result.expectation_terms.value_or(std::vector<Complex>{})};
}

EngineExecutionRequest disk_request(const Circuit& circuit,
                                    std::span<const std::pair<QubitId, ClbitId>> targets,
                                    std::uint64_t shots,
                                    const Observable* observable,
                                    double global_phase,
                                    std::uint32_t threads,
                                    std::uint32_t chunk_qubits,
                                    std::uint32_t window_qubits,
                                    const ExecutionMonitor* monitor = nullptr) {
    return EngineExecutionRequest{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = false,
        .want_counts = shots > 0,
        .want_expectation = observable != nullptr,
        .shots = shots,
        .measure_targets = targets,
        .bitstring_width = circuit.num_clbits(),
        .seed = 7,
        .global_phase = global_phase,
        .budget_bytes = std::numeric_limits<std::uint64_t>::max(),
        .is_sole_region = true,
        .observable = observable,
        .monitor = monitor,
        .max_threads = threads,
        .scratch_directory = std::filesystem::current_path(),
        .tier_chunk_qubits = chunk_qubits,
        .tier_window_qubits = window_qubits,
    };
}

Outputs run_on_disk(const Circuit& circuit,
                    std::uint64_t shots,
                    const Observable* observable,
                    double global_phase,
                    std::uint32_t threads,
                    std::uint32_t chunk_qubits,
                    std::uint32_t window_qubits) {
    const std::vector<std::pair<QubitId, ClbitId>> targets = all_targets(circuit.num_qubits());
    const DiskDenseEngine engine;
    EngineExecutionResult result = engine.execute(disk_request(
        circuit, targets, shots, observable, global_phase, threads, chunk_qubits, window_qubits));
    return Outputs{result.counts.value_or(std::map<BitString, std::uint64_t>{}),
                   result.expectation_terms.value_or(std::vector<Complex>{})};
}

bool same_bits(Complex a, Complex b) {
    return std::bit_cast<std::uint64_t>(a.real()) == std::bit_cast<std::uint64_t>(b.real()) &&
           std::bit_cast<std::uint64_t>(a.imag()) == std::bit_cast<std::uint64_t>(b.imag());
}

void expect_same_outputs(const Outputs& expected, const Outputs& actual, const std::string& where) {
    EXPECT_EQ(actual.counts, expected.counts) << where;
    ASSERT_EQ(actual.terms.size(), expected.terms.size()) << where;
    for (std::size_t i = 0; i < expected.terms.size(); ++i) {
        EXPECT_TRUE(same_bits(actual.terms[i], expected.terms[i]))
            << where << " term " << i << ": " << actual.terms[i] << " vs " << expected.terms[i];
    }
}

TEST(DiskDenseEngineTest, CountsAndExpectationsAreBitwiseTheInMemoryOnesForEveryChunkSize) {
    std::mt19937_64 rng(2306);
    for (const std::uint32_t n : {8U, 10U, 12U}) {
        const Circuit circuit = random_circuit(n, 60, 100 + n);
        const Observable observable = random_observable(n, 5, rng);
        const Outputs expected = run_in_memory(circuit, 2000, &observable, 0.3, 1);
        for (std::uint32_t chunk = 2; chunk <= n; chunk += 2) {
            // The smallest window that fits the gates, and one that holds
            // five global qubits, the widest gate's operand count.
            for (const std::uint32_t window : {0U, std::min(n, chunk + 5)}) {
                const Outputs actual =
                    run_on_disk(circuit, 2000, &observable, 0.3, 1, chunk, window);
                expect_same_outputs(expected,
                                    actual,
                                    "n=" + std::to_string(n) + " L=" + std::to_string(chunk) +
                                        " W=" + std::to_string(window));
            }
        }
    }
}

TEST(DiskDenseEngineTest, EveryThreadCountGivesTheSameBytes) {
    std::mt19937_64 rng(2307);
    const std::uint32_t n = 14;
    const Circuit circuit = random_circuit(n, 40, 214);
    const Observable observable = random_observable(n, 3, rng);
    const Outputs expected = run_in_memory(circuit, 500, &observable, 0.0, 1);
    for (const std::uint32_t threads : {1U, 2U, 3U, common::default_thread_count()}) {
        const Outputs actual = run_on_disk(circuit, 500, &observable, 0.0, threads, 8, 13);
        expect_same_outputs(expected, actual, "threads=" + std::to_string(threads));
    }
}

TEST(DiskDenseEngineTest, GatesOnGlobalQubitsOnlyAndStraddlingGates) {
    // n = 10, L = 4: every gate below touches a global qubit, several both.
    Circuit circuit(10, 10);
    circuit.append(Opcode::H, {9});
    circuit.append(Opcode::CX, {9, 4});
    circuit.append(Opcode::CX, {4, 0});  // straddling
    circuit.append(Opcode::CCX, {9, 8, 7});
    circuit.append(Opcode::RX, {6}, {0.7});
    circuit.append(Opcode::CP, {5, 3}, {1.1});  // straddling
    circuit.append(Opcode::SWAP, {8, 2});       // straddling
    circuit.append(Opcode::CCX, {1, 2, 9});     // two local, one global
    circuit.append(Opcode::CZ, {7, 5});
    circuit.append(Opcode::T, {4});
    for (QubitId q = 0; q < 10; ++q) {
        circuit.append_measure(q, q);
    }
    std::mt19937_64 rng(2308);
    const Observable observable = random_observable(10, 4, rng);
    const Outputs expected = run_in_memory(circuit, 1000, &observable, 0.0, 1);
    for (const std::uint32_t window : {0U, 7U, 10U}) {
        expect_same_outputs(expected,
                            run_on_disk(circuit, 1000, &observable, 0.0, 1, 4, window),
                            "W=" + std::to_string(window));
    }
    // Every gate global with a window of one chunk per global qubit: one pass
    // per gate.
    Circuit global_only(8, 8);
    for (QubitId q = 4; q < 8; ++q) {
        global_only.append(Opcode::H, {q});
        global_only.append(Opcode::RX, {q}, {0.4});
    }
    global_only.append(Opcode::CX, {7, 4});
    global_only.append(Opcode::CX, {5, 6});
    for (QubitId q = 0; q < 8; ++q) {
        global_only.append_measure(q, q);
    }
    const DiskDenseEngine engine;
    const std::vector<Operation>& ops = global_only.operations();
    const EngineEstimateRequest estimate_request{
        .num_qubits = 8,
        .operations = &ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .observable = nullptr,
        .shots = 300,
        .bitstring_width = 8,
        .is_sole_region = true,
        .tier_chunk_qubits = 4,
        .tier_window_qubits = 6,
        // The pass count below is the gate-by-gate plan's; the runs that
        // follow compare the fused default.
        .fuse_gates = false,
    };
    const CostEstimate estimate = engine.estimate(estimate_request);
    ASSERT_TRUE(estimate.feasible) << estimate.infeasible_reason;
    // w = 2: {4, 5} | {6, 7} | {7, 4} | {5, 6}, the measurements in the last.
    EXPECT_EQ(estimate.tier.passes, 4U);
    expect_same_outputs(run_in_memory(global_only, 300, nullptr, 0.0, 1),
                        run_on_disk(global_only, 300, nullptr, 0.0, 1, 4, 6),
                        "global only");
}

TEST(DiskDenseEngineTest, EstimateReportsThePlanAndBoundsTheLedger) {
    const std::uint32_t n = 12;
    const Circuit circuit = random_circuit(n, 50, 312);
    std::mt19937_64 rng(2309);
    const Observable observable = random_observable(n, 2, rng);
    const std::vector<Operation>& ops = circuit.operations();
    const DiskDenseEngine engine;
    const EngineEstimateRequest estimate_request{
        .num_qubits = n,
        .operations = &ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = true,
        .observable = &observable,
        .shots = 1000,
        .bitstring_width = n,
        .is_sole_region = true,
        .tier_chunk_qubits = 6,
        .tier_window_qubits = 11,
    };
    const CostEstimate estimate = engine.estimate(estimate_request);
    ASSERT_TRUE(estimate.feasible) << estimate.infeasible_reason;
    EXPECT_EQ(estimate.tier.chunk_qubits, 6U);
    EXPECT_EQ(estimate.tier.window_qubits, 11U);
    EXPECT_EQ(estimate.tier.file_bytes, (std::uint64_t{1} << n) * sizeof(Complex));
    EXPECT_GT(estimate.tier.passes, 1U);
    // The initial write, two per pass, the sampling read, one or two per term.
    EXPECT_GE(estimate.tier.link_bytes,
              estimate.tier.file_bytes * (2 + 2 * estimate.tier.passes + observable.terms.size()));
    EXPECT_LE(
        estimate.tier.link_bytes,
        estimate.tier.file_bytes * (2 + 2 * estimate.tier.passes + 2 * observable.terms.size()));
    // The same work units as the in-memory engine.
    EngineEstimateRequest in_memory_request = estimate_request;
    in_memory_request.tier_chunk_qubits = 0;
    in_memory_request.tier_window_qubits = 0;
    EXPECT_EQ(estimate.estimated_work_units,
              dense::DenseEngine().estimate(in_memory_request).estimated_work_units);

    common::MemoryLedger ledger;
    {
        const common::MemoryLedger::Scope scope(ledger);
        const std::vector<std::pair<QubitId, ClbitId>> targets = all_targets(n);
        const EngineExecutionResult result =
            engine.execute(disk_request(circuit, targets, 1000, &observable, 0.0, 2, 6, 11));
        EXPECT_EQ(result.actual_bytes, Window::heap_bytes(11) - common::kHeapBlockOverheadBytes);
    }
    EXPECT_LE(ledger.peak_host_bytes(), estimate.estimated_peak_bytes);
    EXPECT_GE(ledger.peak_host_bytes(), (std::uint64_t{1} << 11) * sizeof(Complex));
    EXPECT_EQ(ledger.peak_disk_bytes(), estimate.tier.file_bytes);
    EXPECT_EQ(ledger.live_disk_bytes(), 0U);
    EXPECT_EQ(ledger.live_host_bytes(), 0U);
}

TEST(DiskDenseEngineTest, TheEngineChoosesTheWindowFromTheBudgetAndTheChunkFromThePasses) {
    const std::uint32_t n = 12;
    const Circuit circuit = random_circuit(n, 50, 412);
    const std::vector<Operation>& ops = circuit.operations();
    const DiskDenseEngine engine;
    EngineEstimateRequest request{
        .num_qubits = n,
        .operations = &ops,
        .want_statevector = false,
        .want_counts = true,
        .want_expectation = false,
        .observable = nullptr,
        .shots = 100,
        .bitstring_width = n,
        .is_sole_region = true,
    };
    // An unbounded budget: the whole state is one window and one chunk.
    const CostEstimate whole = engine.estimate(request);
    ASSERT_TRUE(whole.feasible) << whole.infeasible_reason;
    EXPECT_EQ(whole.tier.window_qubits, n);
    EXPECT_EQ(whole.tier.chunk_qubits, n);
    EXPECT_EQ(whole.tier.passes, 1U);
    // A budget three qubits of window short of that peak.
    request.budget_bytes =
        whole.estimated_peak_bytes - (Window::heap_bytes(n) - Window::heap_bytes(n - 3));
    const CostEstimate smaller = engine.estimate(request);
    ASSERT_TRUE(smaller.feasible) << smaller.infeasible_reason;
    EXPECT_EQ(smaller.tier.window_qubits, n - 3);
    EXPECT_LE(smaller.estimated_peak_bytes, request.budget_bytes);
    EXPECT_GT(smaller.tier.passes, 1U);
    EXPECT_LT(smaller.tier.chunk_qubits, n - 3);  // the fewest passes need global room
    // The run derives the same layout from the estimate's own peak as its
    // budget, as the runtime hands it over.
    const std::vector<std::pair<QubitId, ClbitId>> targets = all_targets(n);
    EngineExecutionRequest run = disk_request(circuit, targets, 100, nullptr, 0.0, 1, 0, 0);
    run.budget_bytes = smaller.estimated_peak_bytes;
    const EngineExecutionResult result = engine.execute(run);
    EXPECT_EQ(result.actual_bytes, Window::heap_bytes(n - 3) - common::kHeapBlockOverheadBytes);
    ASSERT_TRUE(result.counts.has_value());
    EXPECT_EQ(result.counts.value_or(std::map<BitString, std::uint64_t>{}),
              run_in_memory(circuit, 100, nullptr, 0.0, 1).counts);
    // A budget below the smallest window is refused with the figures.
    request.budget_bytes = 1;
    const CostEstimate none = engine.estimate(request);
    EXPECT_FALSE(none.feasible);
    EXPECT_NE(none.infeasible_reason.find("exceeds the budget of 1 bytes"), std::string::npos)
        << none.infeasible_reason;
}

TEST(DiskDenseEngineTest, RefusesWhatItCannotDoWithTheReason) {
    const Circuit circuit = random_circuit(8, 10, 508);
    const std::vector<Operation>& ops = circuit.operations();
    const DiskDenseEngine engine;
    EngineEstimateRequest request{
        .num_qubits = 8,
        .operations = &ops,
        .want_statevector = true,
        .want_counts = false,
        .want_expectation = false,
        .observable = nullptr,
        .is_sole_region = true,
    };
    const CostEstimate statevector = engine.estimate(request);
    EXPECT_FALSE(statevector.feasible);
    EXPECT_NE(statevector.infeasible_reason.find("does not return a statevector"),
              std::string::npos)
        << statevector.infeasible_reason;
    request.want_statevector = false;
    request.is_sole_region = false;
    const CostEstimate region = engine.estimate(request);
    EXPECT_FALSE(region.feasible);
    EXPECT_NE(region.infeasible_reason.find("sole region"), std::string::npos)
        << region.infeasible_reason;
    request.is_sole_region = true;
    // Two global operands in a window of one chunk.
    Circuit wide(8, 0);
    wide.append(Opcode::CX, {6, 7});
    const std::vector<Operation>& wide_ops = wide.operations();
    request.operations = &wide_ops;
    request.tier_chunk_qubits = 4;
    request.tier_window_qubits = 4;
    const CostEstimate too_small = engine.estimate(request);
    EXPECT_FALSE(too_small.feasible);
    EXPECT_NE(too_small.infeasible_reason.find("a gate has 2 operands above the chunk"),
              std::string::npos)
        << too_small.infeasible_reason;
    // A qubit count beyond the dense ceiling is refused for its size before
    // any pass plan is built (a 100-qubit mask would not fit a word).
    Circuit huge(100, 0);
    huge.append(Opcode::H, {99});
    const std::vector<Operation>& huge_ops = huge.operations();
    request.operations = &huge_ops;
    request.num_qubits = 100;
    request.tier_chunk_qubits = 0;
    request.tier_window_qubits = 0;
    const CostEstimate beyond = engine.estimate(request);
    EXPECT_FALSE(beyond.feasible);
    EXPECT_NE(beyond.infeasible_reason.find("checked_pow2"), std::string::npos)
        << beyond.infeasible_reason;
    EXPECT_EQ(engine.capabilities().statevector, false);
    EXPECT_EQ(engine.tier(), Tier::Disk);
    EXPECT_EQ(engine.engine_id(), Engine::Dense);
    EXPECT_EQ(engine.backend(), Backend::Cpu);
}

class ThrowAtPass : public ExecutionMonitor {
public:
    explicit ThrowAtPass(std::uint64_t boundaries_to_allow) : allow_(boundaries_to_allow) {}
    void at_safe_boundary(std::uint64_t boundaries_crossed) const override {
        seen_.push_back(boundaries_crossed);
        if (seen_.size() > allow_) {
            throw std::runtime_error("abandoned at a pass boundary");
        }
    }
    [[nodiscard]] const std::vector<std::uint64_t>& seen() const {
        return seen_;
    }

private:
    std::uint64_t allow_;
    mutable std::vector<std::uint64_t> seen_;
};

std::size_t scratch_files_of_this_process() {
#if defined(_WIN32)
    const auto pid = static_cast<unsigned long>(_getpid());
#else
    const auto pid = static_cast<unsigned long>(getpid());
#endif
    const std::string prefix = "menos-scratch-" + std::to_string(pid) + "-";
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::current_path())) {
        if (entry.path().filename().string().starts_with(prefix)) {
            ++count;
        }
    }
    return count;
}

TEST(DiskDenseEngineTest, AMonitorThatThrowsAbandonsAfterAPassAndTheFileIsGone) {
    const std::uint32_t n = 10;
    const Circuit circuit = random_circuit(n, 40, 610);
    const std::vector<std::pair<QubitId, ClbitId>> targets = all_targets(n);
    const DiskDenseEngine engine;
    const std::size_t before = scratch_files_of_this_process();
    const ThrowAtPass monitor(1);
    // Chunks of 2^4 in the smallest window that fits the gates.
    EXPECT_THROW(
        (void)engine.execute(disk_request(circuit, targets, 100, nullptr, 0.0, 1, 4, 0, &monitor)),
        std::runtime_error);
    // Called once per pass with the blocks applied so far (Phase 24 item 7:
    // the pass's units are the fusion plan's blocks), increasing.
    const std::size_t blocks = ir::plan_fusion(circuit.operations()).block_count();
    ASSERT_EQ(monitor.seen().size(), 2U);
    EXPECT_LT(monitor.seen()[0], monitor.seen()[1]);
    EXPECT_LE(monitor.seen()[1], blocks);
    EXPECT_EQ(scratch_files_of_this_process(), before);
    // The next run is correct.
    expect_same_outputs(run_in_memory(circuit, 100, nullptr, 0.0, 1),
                        run_on_disk(circuit, 100, nullptr, 0.0, 1, 4, 0),
                        "after abandon");
    // A monitor that never throws sees every pass end, the last at the end.
    const ThrowAtPass patient(1'000'000);
    const EngineExecutionResult result =
        engine.execute(disk_request(circuit, targets, 100, nullptr, 0.0, 1, 4, 0, &patient));
    EXPECT_FALSE(patient.seen().empty());
    EXPECT_EQ(patient.seen().back(), blocks);
    EXPECT_TRUE(result.counts.has_value());
}

TEST(DiskDenseEngineTest, DepositBitsPlacesValueBitsAtTheMaskBits) {
    EXPECT_EQ(deposit_bits(0b101, 0b1110), 0b1010U);
    EXPECT_EQ(deposit_bits(0b11, 0b1001), 0b1001U);
    EXPECT_EQ(deposit_bits(0, 0b1111), 0U);
    EXPECT_EQ(deposit_bits(0b1111, 0), 0U);
    EXPECT_EQ(deposit_bits(1, std::uint64_t{1} << 63), std::uint64_t{1} << 63);
}

}  // namespace
}  // namespace menos::engines::dense_tiered
