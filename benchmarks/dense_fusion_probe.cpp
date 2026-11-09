// Phase 24 step 1, item 7 (docs/adr/0028 section 17): is the number of passes
// over the state what limits the dense CPU engine's speed, and what would
// fusing neighbouring gates into one block buy?
//
// Two measurements, both on the unchanged kernels (kernels::cpu::apply_matrix
// through engines::dense::apply_matrix), no fusion code in the library:
//
//   1. The cost of ONE pass by operand count k = 1..5 and state size n, on one
//      thread and on every hardware thread: seconds per call, and the bytes
//      the pass streams (the state read and written once, 2 * 16 * 2^n) over
//      that time against the calibration's own streaming bandwidth. Where the
//      per-pass time is the same for every k and close to the bandwidth
//      floor, a pass is bandwidth-bound and fewer passes is the only lever.
//      Where it grows with k, the arithmetic per amplitude is what a fused
//      block has to pay for.
//
//   2. A random brickwork circuit (U on every qubit, then CX on a random
//      perfect matching, per layer) applied gate by gate as the engine does,
//      and applied as blocks fused offline in this program, for windows
//      k = 2..5: the pass count, the time, the speed-up, and the largest
//      amplitude difference between the two final states (the rounding that
//      fusion introduces). The grouping is the simplest sound rule: a gate
//      joins the oldest open block it fits in that it can commute back to
//      (every block between them is on disjoint qubits), else opens a new
//      block. It is a measurement device, not the design; the design is the
//      ADR's.
//
//   3. (gate run, docs/adr/0028 section 17.7) The same brickwork circuits
//      through the public menos::run with RunOptions::fuse_gates on and off:
//      execute_seconds, their same-run ratio, the plan's fusion outcome and
//      the largest amplitude difference between the two statevectors. Rows
//      "run".
//
// Usage: menos_dense_fusion_probe [max_qubits]
// CSV on stdout, the environment on stderr. A report, not a unit test.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "benchmark_spread.h"
#include "common/checked_math.h"
#include "common/thread_count.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/state_vector.h"
#include "ir/gate_table.h"
#include "menos/circuit.h"
#include "menos/menos.h"
#include "menos_benchmark_environment.h"
#include "plan/hardware_profile.h"

namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;
using menos::QubitId;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;

// Repetitions per timed point (owner, Phase 15 Q1: 10 per point) for the
// single-pass measurement; the circuit measurement repeats fewer times
// because one unfused repetition at 24 qubits is tens of seconds.
constexpr int kPassRepetitions = 10;
constexpr int kCircuitRepetitions = 3;
constexpr int kCircuitDepth = 10;
constexpr std::uint32_t kMaxWindow = 5;  // kMaxUnitaryQubits, the generic kernel's ceiling

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<Complex> random_matrix(std::size_t dim, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> m(dim * dim);
    for (Complex& c : m) {
        c = Complex{dist(rng), dist(rng)};
    }
    return m;
}

// Operands for a k-qubit gate spread across the register, so no two sit in
// the same low run and the kernels take their general paths.
std::vector<std::uint32_t> spread_operands(std::uint32_t n, std::uint32_t k) {
    std::vector<std::uint32_t> operands;
    if (k == 1) {
        operands.push_back(n / 2);
        return operands;
    }
    for (std::uint32_t j = 0; j < k; ++j) {
        operands.push_back(
            static_cast<std::uint32_t>((static_cast<std::uint64_t>(j) * (n - 1)) / (k - 1)));
    }
    return operands;
}

bool fits(std::uint32_t n, std::uint64_t budget) {
    return menos::common::checked_mul(menos::common::checked_pow2(n), sizeof(Complex)) <= budget;
}

menos::engines::dense::StateVector fresh_state(std::uint32_t n) {
    menos::engines::dense::StateVector sv(
        static_cast<int>(n),
        menos::common::checked_mul(menos::common::checked_pow2(n), sizeof(Complex)));
    menos::engines::dense::initialize(sv, 0.0);
    return sv;
}

// --- 1. one pass by k and n -------------------------------------------------

// A diagonal k-qubit matrix of random phases: the streaming kernels' case.
std::vector<Complex> random_diagonal(std::size_t dim, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    std::vector<Complex> m(dim * dim, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < dim; ++i) {
        m[i * dim + i] = std::polar(1.0, angle(rng));
    }
    return m;
}

// A qubit count, a byte budget and a bandwidth: three roles, named at the
// one call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void measure_passes(std::uint32_t max_qubits, std::uint64_t budget, double bandwidth) {
    const std::uint32_t all = menos::common::default_thread_count();
    std::mt19937_64 rng(2401);
    std::printf(
        "kind,qubits,k,threads,median_s,min_s,max_s,bytes_per_s,bandwidth_floor_s,"
        "amplitudes_per_s\n");
    for (std::uint32_t n = 14; n <= max_qubits; n += 2) {
        if (!fits(n, budget)) {
            break;
        }
        menos::engines::dense::StateVector sv = fresh_state(n);
        const double bytes = 2.0 * 16.0 * std::ldexp(1.0, static_cast<int>(n));
        const auto time_pass = [&](const char* kind,
                                   std::uint32_t k,
                                   std::uint32_t threads,
                                   const std::vector<Complex>& matrix) {
            const std::vector<std::uint32_t> operands = spread_operands(n, k);
            menos::engines::dense::apply_matrix(sv.amplitudes(), matrix, operands, threads);
            std::vector<double> samples;
            for (int r = 0; r < kPassRepetitions; ++r) {
                const Clock::time_point start = Clock::now();
                menos::engines::dense::apply_matrix(sv.amplitudes(), matrix, operands, threads);
                samples.push_back(seconds_since(start));
            }
            const Spread s = spread(samples);
            std::printf("%s,%u,%u,%u,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        kind,
                        n,
                        k,
                        threads,
                        s.median,
                        s.min,
                        s.max,
                        bytes / s.median,
                        bandwidth > 0.0 ? bytes / bandwidth : 0.0,
                        std::ldexp(1.0, static_cast<int>(n)) / s.median);
            std::fflush(stdout);
        };
        for (const std::uint32_t threads : {1U, all}) {
            for (std::uint32_t k = 1; k <= kMaxWindow; ++k) {
                time_pass("pass_dense", k, threads, random_matrix(std::size_t{1} << k, rng));
            }
            for (std::uint32_t k = 1; k <= kMaxWindow; ++k) {
                time_pass("pass_diagonal", k, threads, random_diagonal(std::size_t{1} << k, rng));
            }
            if (threads == all) {
                break;  // a machine with one hardware thread
            }
        }
    }
}

// --- 2. a circuit, gate by gate and in fused blocks ---------------------------

struct Gate {
    std::vector<QubitId> qubits;
    std::vector<Complex> matrix;  // generated once, as the engine generates it per gate
};

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): n qubits, depth layers.
std::vector<Gate> brickwork(std::uint32_t n, int depth, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(0.0, 6.283185307179586);
    std::vector<Gate> gates;
    std::vector<QubitId> order(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        order[q] = q;
    }
    for (int layer = 0; layer < depth; ++layer) {
        for (QubitId q = 0; q < n; ++q) {
            const std::vector<double> params = {angle(rng), angle(rng), angle(rng)};
            gates.push_back(Gate{{q}, menos::ir::generate_matrix(menos::Opcode::U, params)});
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (std::uint32_t i = 0; i + 1 < n; i += 2) {
            gates.push_back(
                Gate{{order[i], order[i + 1]}, menos::ir::generate_matrix(menos::Opcode::CX, {})});
        }
    }
    return gates;
}

// Applies gate `g` to the 2^k-vector v whose bit j is block qubit `block[j]`:
// a plain gather/matrix/scatter, the oracle of tests/ir/fuse_diagonal_test.cpp.
void apply_to_vector(std::vector<Complex>& v, const Gate& g, const std::vector<QubitId>& block) {
    std::vector<std::uint32_t> local(g.qubits.size());
    for (std::size_t j = 0; j < g.qubits.size(); ++j) {
        local[j] = static_cast<std::uint32_t>(std::find(block.begin(), block.end(), g.qubits[j]) -
                                              block.begin());
    }
    const std::size_t dim = std::size_t{1} << g.qubits.size();
    std::vector<Complex> next(v.size());
    for (std::size_t m = 0; m < v.size(); ++m) {
        std::size_t row = 0;
        for (std::size_t j = 0; j < local.size(); ++j) {
            row |= ((m >> local[j]) & std::size_t{1}) << j;
        }
        Complex sum{0.0, 0.0};
        for (std::size_t in = 0; in < dim; ++in) {
            std::size_t src = m;
            for (std::size_t j = 0; j < local.size(); ++j) {
                const std::size_t bit = std::size_t{1} << local[j];
                src = ((in >> j) & 1U) != 0 ? (src | bit) : (src & ~bit);
            }
            sum += g.matrix[row * dim + in] * v[src];
        }
        next[m] = sum;
    }
    v = std::move(next);
}

struct Block {
    std::vector<QubitId> qubits;  // in the order they joined; sorted when the matrix is built
    std::vector<const Gate*> gates;
};

// Both symmetric in their two qubit lists.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
bool intersects(const std::vector<QubitId>& a, const std::vector<QubitId>& b) {
    return std::any_of(
        a.begin(), a.end(), [&](QubitId q) { return std::find(b.begin(), b.end(), q) != b.end(); });
}

std::size_t union_size(const std::vector<QubitId>& a, const std::vector<QubitId>& b) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::size_t extra = 0;
    for (const QubitId q : b) {
        if (std::find(a.begin(), a.end(), q) == a.end()) {
            ++extra;
        }
    }
    return a.size() + extra;
}

// The grouping rule described at the top of the file. Blocks are emitted in
// the order they opened, which is sound: a gate only ever joins a block at or
// after its newest dependency, and every block opened after the one it joins
// is on qubits disjoint from the gate's.
std::vector<Block> fuse(const std::vector<Gate>& gates, std::uint32_t window) {
    std::vector<Block> blocks;
    for (const Gate& g : gates) {
        std::size_t start = 0;
        for (std::size_t i = blocks.size(); i-- > 0;) {
            if (intersects(blocks[i].qubits, g.qubits)) {
                start = i;
                break;
            }
        }
        bool joined = false;
        for (std::size_t i = start; i < blocks.size(); ++i) {
            if (union_size(blocks[i].qubits, g.qubits) <= window) {
                for (const QubitId q : g.qubits) {
                    if (std::find(blocks[i].qubits.begin(), blocks[i].qubits.end(), q) ==
                        blocks[i].qubits.end()) {
                        blocks[i].qubits.push_back(q);
                    }
                }
                blocks[i].gates.push_back(&g);
                joined = true;
                break;
            }
        }
        if (!joined) {
            blocks.push_back(Block{g.qubits, {&g}});
        }
    }
    return blocks;
}

// One Gate per block: the product of its gates, in block order, on the
// block's ascending qubit list (the engine's operand-order rule).
Gate block_gate(const Block& block) {
    std::vector<QubitId> qubits = block.qubits;
    std::sort(qubits.begin(), qubits.end());
    const std::size_t dim = std::size_t{1} << qubits.size();
    std::vector<Complex> matrix(dim * dim, Complex{0.0, 0.0});
    for (std::size_t col = 0; col < dim; ++col) {
        std::vector<Complex> column(dim, Complex{0.0, 0.0});
        column[col] = Complex{1.0, 0.0};
        for (const Gate* g : block.gates) {
            apply_to_vector(column, *g, qubits);
        }
        for (std::size_t row = 0; row < dim; ++row) {
            matrix[row * dim + col] = column[row];
        }
    }
    return Gate{std::move(qubits), std::move(matrix)};
}

struct CircuitTiming {
    Spread seconds;
    std::vector<Complex> final_state;
};

// A thread count and a repetition count: distinct roles, named at every call.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
CircuitTiming time_circuit(std::uint32_t n,
                           const std::vector<Gate>& gates,
                           std::uint32_t threads,
                           int repetitions) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    std::vector<double> samples;
    std::vector<Complex> final_state;
    for (int r = -1; r < repetitions; ++r) {  // r = -1 is the warm-up
        menos::engines::dense::StateVector sv = fresh_state(n);
        const Clock::time_point start = Clock::now();
        for (const Gate& g : gates) {
            menos::engines::dense::apply_matrix(sv.amplitudes(), g.matrix, g.qubits, threads);
        }
        const double seconds = seconds_since(start);
        if (r >= 0) {
            samples.push_back(seconds);
        }
        if (r == repetitions - 1) {
            final_state = std::move(sv).release();
        }
    }
    return CircuitTiming{spread(samples), std::move(final_state)};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): symmetric.
double max_difference(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

void measure_circuits(std::uint32_t max_qubits, std::uint64_t budget) {
    const std::uint32_t all = menos::common::default_thread_count();
    std::mt19937_64 rng(2407);
    std::printf(
        "kind,qubits,window,threads,gates,passes,gates_absorbed,median_s,min_s,max_s,"
        "speedup,max_abs_diff\n");
    for (const std::uint32_t n : {16U, 20U, 22U, 24U}) {
        if (n > max_qubits || !fits(n, budget)) {
            break;
        }
        const std::vector<Gate> gates = brickwork(n, kCircuitDepth, rng);
        for (const std::uint32_t threads : {1U, all}) {
            if (threads == 1 && n > 20) {
                continue;  // one unfused repetition at 22+ qubits on one thread is minutes
            }
            const CircuitTiming unfused = time_circuit(n, gates, threads, kCircuitRepetitions);
            std::printf("circuit,%u,1,%u,%zu,%zu,0,%.6e,%.6e,%.6e,1,0\n",
                        n,
                        threads,
                        gates.size(),
                        gates.size(),
                        unfused.seconds.median,
                        unfused.seconds.min,
                        unfused.seconds.max);
            std::fflush(stdout);
            for (std::uint32_t window = 2; window <= kMaxWindow; ++window) {
                const std::vector<Block> blocks = fuse(gates, window);
                std::vector<Gate> fused;
                fused.reserve(blocks.size());
                for (const Block& block : blocks) {
                    fused.push_back(block_gate(block));
                }
                const CircuitTiming timing = time_circuit(n, fused, threads, kCircuitRepetitions);
                std::printf("circuit,%u,%u,%u,%zu,%zu,%zu,%.6e,%.6e,%.6e,%.4f,%.3e\n",
                            n,
                            window,
                            threads,
                            gates.size(),
                            fused.size(),
                            gates.size() - fused.size(),
                            timing.seconds.median,
                            timing.seconds.min,
                            timing.seconds.max,
                            unfused.seconds.median / timing.seconds.median,
                            max_difference(unfused.final_state, timing.final_state));
                std::fflush(stdout);
            }
            if (threads == all) {
                break;
            }
        }
    }
}

// --- 3. the engine itself, fused and unfused, through menos::run -------------

menos::Circuit brickwork_circuit(std::uint32_t n, int depth, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(0.0, 6.283185307179586);
    menos::Circuit circuit(n, 0);
    std::vector<QubitId> order(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        order[q] = q;
    }
    for (int layer = 0; layer < depth; ++layer) {
        for (QubitId q = 0; q < n; ++q) {
            circuit.append(menos::Opcode::U, {q}, {angle(rng), angle(rng), angle(rng)});
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (std::uint32_t i = 0; i + 1 < n; i += 2) {
            circuit.append(menos::Opcode::CX, {order[i], order[i + 1]});
        }
    }
    return circuit;
}

struct RunTiming {
    Spread seconds;
    std::vector<Complex> statevector;
    std::uint64_t blocks = 0;
    std::uint64_t absorbed = 0;
};

RunTiming time_run(const menos::Circuit& circuit, bool fuse, std::uint32_t threads) {
    menos::RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
    options.max_threads = threads;
    options.fuse_gates = fuse;
    options.backend = menos::Backend::Cpu;
    std::vector<double> samples;
    RunTiming timing;
    for (int r = -1; r < kCircuitRepetitions; ++r) {
        menos::Result result = menos::run(circuit, options);
        if (r >= 0) {
            samples.push_back(result.execute_seconds());
        }
        if (r == kCircuitRepetitions - 1) {
            timing.statevector = *result.statevector();
            timing.blocks = result.plan().regions[0].fusion.fused_blocks;
            timing.absorbed = result.plan().regions[0].fusion.gates_absorbed;
        }
    }
    timing.seconds = spread(samples);
    return timing;
}

void measure_runs(std::uint32_t max_qubits, std::uint64_t budget) {
    const std::uint32_t all = menos::common::default_thread_count();
    std::mt19937_64 rng(2411);
    std::printf(
        "kind,qubits,threads,gates,fused_blocks,gates_absorbed,unfused_median_s,unfused_min_s,"
        "unfused_max_s,fused_median_s,fused_min_s,fused_max_s,speedup,max_abs_diff\n");
    for (const std::uint32_t n : {16U, 20U, 22U, 24U}) {
        if (n > max_qubits || !fits(n, budget)) {
            break;
        }
        const menos::Circuit circuit = brickwork_circuit(n, kCircuitDepth, rng);
        for (const std::uint32_t threads : {1U, all}) {
            if (threads == 1 && n > 20) {
                continue;
            }
            const RunTiming unfused = time_run(circuit, /*fuse=*/false, threads);
            const RunTiming fused = time_run(circuit, /*fuse=*/true, threads);
            std::printf("run,%u,%u,%zu,%llu,%llu,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%.3e\n",
                        n,
                        threads,
                        circuit.operations().size(),
                        static_cast<unsigned long long>(fused.blocks),
                        static_cast<unsigned long long>(fused.absorbed),
                        unfused.seconds.median,
                        unfused.seconds.min,
                        unfused.seconds.max,
                        fused.seconds.median,
                        fused.seconds.min,
                        fused.seconds.max,
                        unfused.seconds.median / fused.seconds.median,
                        max_difference(unfused.statevector, fused.statevector));
            std::fflush(stdout);
            if (threads == all) {
                break;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t max_qubits =
            argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 24U;
        const std::uint64_t budget = menos_benchmark::safe_allocation_budget_bytes();
        const menos::plan::HardwareProfile& profile = menos::plan::calibrated_hardware_profile();
        std::fprintf(stderr,
                     "%s\nthreads=%u memory_bandwidth_B_per_s=%.6e work_units_per_s=%.6e "
                     "safe_budget_bytes=%llu\n",
                     menos_benchmark::describe_environment().c_str(),
                     menos::common::default_thread_count(),
                     profile.memory_bandwidth_bytes_per_second,
                     profile.work_units_per_second,
                     static_cast<unsigned long long>(budget));
        measure_passes(max_qubits, budget, profile.memory_bandwidth_bytes_per_second);
        measure_circuits(max_qubits, budget);
        measure_runs(max_qubits, budget);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dense_fusion_probe: %s\n", e.what());
        return 1;
    }
}
