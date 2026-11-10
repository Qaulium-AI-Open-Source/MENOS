// Phase 24 item 8, step 1 (docs/adr/0028 section 19.3): which dense CPU
// kernels are bound by arithmetic, where SIMD can help, and which already run
// at memory bandwidth, where it cannot?
//
// Table A: the cost of ONE pass of each kernel kernels::cpu::apply_matrix
// dispatches to, by the operand placement that selects its code path:
//
//   1q        operands 0, 2 (pair_blocks), 3, n/2, n-1 (pair_run)
//   2q        {0,1}, {1,0}, {0,n-1}, {n/2-1,n/2}, {n-2,n-1}
//   diag1     {0} (scale_elements), {n-1} (scale_run)
//   diag2     {0,1} (scale_elements), {n-2,n-1} (scale_run)
//   generic3  {0,n/2,n-1} (apply_matrix_generic, the oracle)
//
// at n = 16, 20, 22, 24, on one thread and on every hardware thread: the
// median, min and max of kPassRepetitions timed passes after one discarded
// pass, the bytes a pass streams (the state read and written once,
// 2 * 16 * 2^n) over that time, and the calibration's own streaming-bandwidth
// floor for the same bytes. A pass far above its floor is limited by the
// arithmetic per amplitude; a pass near it, by memory.
//
// The same binary measures whichever kernels the build has: the step-1
// "clones off" row set is this probe built from a tree whose
// MENOS_KERNEL_CLONES is empty (docs/benchmarks/phase-24/simd-step1/).
//
// Usage: menos_dense_simd_probe [max_qubits]
// CSV on stdout, the environment on stderr. A report, not a unit test.

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
#include "menos_benchmark_environment.h"
#include "plan/hardware_profile.h"

namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;
using menos::benchmarks::spread;
using menos::benchmarks::Spread;

// Owner, Phase 15 Q1: 10 repetitions per timed point.
constexpr int kPassRepetitions = 10;

struct Case {
    const char* kind;
    std::vector<std::uint32_t> operands;
    bool diagonal;
};

std::vector<Complex> random_matrix(std::size_t dim, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> m(dim * dim);
    for (Complex& c : m) {
        c = Complex{dist(rng), dist(rng)};
    }
    return m;
}

std::vector<Complex> random_diagonal(std::size_t dim, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    std::vector<Complex> m(dim * dim, Complex{0.0, 0.0});
    for (std::size_t i = 0; i < dim; ++i) {
        m[i * dim + i] = std::polar(1.0, angle(rng));
    }
    return m;
}

std::vector<Case> cases_for(std::uint32_t n) {
    const std::uint32_t mid = n / 2;
    return {
        {"1q", {0}, false},
        {"1q", {2}, false},
        {"1q", {3}, false},
        {"1q", {mid}, false},
        {"1q", {n - 1}, false},
        {"2q", {0, 1}, false},
        {"2q", {1, 0}, false},
        {"2q", {0, n - 1}, false},
        {"2q", {mid - 1, mid}, false},
        {"2q", {n - 2, n - 1}, false},
        {"diag1", {0}, true},
        {"diag1", {n - 1}, true},
        {"diag2", {0, 1}, true},
        {"diag2", {n - 2, n - 1}, true},
        {"generic3", {0, mid, n - 1}, false},
    };
}

std::string join(const std::vector<std::uint32_t>& operands) {
    std::string s;
    for (const std::uint32_t q : operands) {
        if (!s.empty()) {
            s += ' ';
        }
        s += std::to_string(q);
    }
    return s;
}

bool fits(std::uint32_t n, std::uint64_t budget) {
    return menos::common::checked_mul(menos::common::checked_pow2(n), sizeof(Complex)) <= budget;
}

// A qubit count, a byte budget and a bandwidth: three roles, named at the
// one call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void measure_kernels(std::uint32_t max_qubits, std::uint64_t budget, double bandwidth) {
    const std::uint32_t all = menos::common::default_thread_count();
    std::mt19937_64 rng(2419);
    std::printf(
        "kind,qubits,operands,threads,median_s,min_s,max_s,bytes_per_s,bandwidth_floor_s,"
        "floor_ratio\n");
    for (const std::uint32_t n : {16U, 20U, 22U, 24U}) {
        if (n > max_qubits || !fits(n, budget)) {
            break;
        }
        menos::engines::dense::StateVector sv(
            static_cast<int>(n),
            menos::common::checked_mul(menos::common::checked_pow2(n), sizeof(Complex)));
        menos::engines::dense::initialize(sv, 0.0);
        const double bytes = 2.0 * 16.0 * std::ldexp(1.0, static_cast<int>(n));
        const double floor_s = bandwidth > 0.0 ? bytes / bandwidth : 0.0;
        for (const std::uint32_t threads : {1U, all}) {
            for (const Case& c : cases_for(n)) {
                const std::size_t dim = std::size_t{1} << c.operands.size();
                const std::vector<Complex> matrix =
                    c.diagonal ? random_diagonal(dim, rng) : random_matrix(dim, rng);
                menos::engines::dense::apply_matrix(sv.amplitudes(), matrix, c.operands, threads);
                std::vector<double> samples;
                for (int r = 0; r < kPassRepetitions; ++r) {
                    const Clock::time_point start = Clock::now();
                    menos::engines::dense::apply_matrix(
                        sv.amplitudes(), matrix, c.operands, threads);
                    samples.push_back(
                        std::chrono::duration<double>(Clock::now() - start).count());
                }
                const Spread s = spread(samples);
                std::printf("%s,%u,%s,%u,%.6e,%.6e,%.6e,%.6e,%.6e,%.3f\n",
                            c.kind,
                            n,
                            join(c.operands).c_str(),
                            threads,
                            s.median,
                            s.min,
                            s.max,
                            bytes / s.median,
                            floor_s,
                            floor_s > 0.0 ? s.median / floor_s : 0.0);
                std::fflush(stdout);
            }
            if (threads == all) {
                break;  // a machine with one hardware thread
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
        measure_kernels(max_qubits, budget, profile.memory_bandwidth_bytes_per_second);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dense_simd_probe: %s\n", e.what());
        return 1;
    }
}
