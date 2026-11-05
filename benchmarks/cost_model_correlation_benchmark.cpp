// Benchmark group 3 (docs/BUILD_PLAN.md Phase 3, 3.9): cost-model work
// units versus measured execution time, reported as a Spearman rank
// correlation. This is evidence that the cost model is USEFUL on this
// machine for ranking candidates, never a proof it predicts wall-clock
// time universally (CLAUDE.md, docs/BUILD_PLAN.md 3.10: "distinguish
// mathematical correctness from empirical performance"). Not a Google
// Benchmark BENCHMARK()-registered program -- correlation is inherently a
// cross-case computation Google Benchmark's per-case timer model doesn't
// fit; this program has its own main() and reuses only this suite's
// shared environment-reporting helper.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "cost/cost_model.h"
#include "engines/dense/dense_engine.h"
#include "menos/menos.h"
#include "menos_benchmark_environment.h"

namespace {

using namespace std::chrono;

menos::Circuit make_brickwork_circuit(std::uint32_t n) {
    menos::Circuit circuit(n, 0);
    for (menos::QubitId q = 0; q < n; ++q) {
        circuit.append(menos::Opcode::H, {q});
    }
    for (menos::QubitId q = 0; q + 1 < n; q += 2) {
        circuit.append(menos::Opcode::CX, {q, static_cast<menos::QubitId>(q + 1)});
    }
    return circuit;
}

double measure_seconds(const menos::Circuit& circuit, int repetitions) {
    const menos::RunOptions options{.shots = 0, .want_statevector = true, .seed = 1};
    // One untimed warm-up run (first-touch page faults, allocator
    // warm-up) before the timed repetitions.
    static_cast<void>(menos::run(circuit, options));

    const auto start = steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        const menos::Result result = menos::run(circuit, options);
        static_cast<void>(result.statevector());
    }
    const auto elapsed = duration<double>(steady_clock::now() - start).count();
    return elapsed / repetitions;
}

// Spearman rank correlation: Pearson correlation of the two rank
// sequences. No ties expected here (n and work_units are both strictly
// increasing across the chosen circuit sizes), so no tie-correction term
// is needed.
double spearman_rank_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    const auto rank_of = [](const std::vector<double>& v) {
        std::vector<std::size_t> order(v.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(
            order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
        std::vector<double> rank(v.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            rank[order[i]] = static_cast<double>(i);
        }
        return rank;
    };
    const std::vector<double> rx = rank_of(x);
    const std::vector<double> ry = rank_of(y);

    const double n = static_cast<double>(x.size());
    const double mean_rx = std::accumulate(rx.begin(), rx.end(), 0.0) / n;
    const double mean_ry = std::accumulate(ry.begin(), ry.end(), 0.0) / n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    for (std::size_t i = 0; i < rx.size(); ++i) {
        const double dx = rx[i] - mean_rx;
        const double dy = ry[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    return cov / std::sqrt(var_x * var_y);
}

}  // namespace

int main() {
    std::cout << menos_benchmark::describe_environment() << "\n";
    std::cout << "cost-model correlation: work_units (direct, brickwork H+CX circuit) vs measured "
                 "execution seconds\n";

    const menos::engines::dense::DenseEngine dense_engine;
    std::vector<double> work_units;
    std::vector<double> seconds;
    for (const std::uint32_t n : {std::uint32_t{10},
                                  std::uint32_t{12},
                                  std::uint32_t{14},
                                  std::uint32_t{16},
                                  std::uint32_t{18},
                                  std::uint32_t{20}}) {
        const menos::Circuit circuit = make_brickwork_circuit(n);
        const auto estimate =
            menos::cost::estimate_direct(circuit,
                                         dense_engine,
                                         /*want_statevector=*/true,
                                         /*want_counts=*/false,
                                         /*want_expectation=*/false,
                                         /*observable=*/nullptr,
                                         menos_benchmark::safe_allocation_budget_bytes(),
                                         /*accuracy_policy=*/{},
                                         /*trajectories=*/0,
                                         /*shots=*/0);
        if (!estimate.feasible) {
            std::cout << "  n=" << n << ": skipped (" << estimate.infeasible_reason << ")\n";
            continue;
        }
        const double t = measure_seconds(circuit, /*repetitions=*/3);
        std::cout << "  n=" << n << " work_units=" << estimate.estimated_work_units
                  << " seconds=" << t << "\n";
        work_units.push_back(static_cast<double>(estimate.estimated_work_units));
        seconds.push_back(t);
    }

    if (work_units.size() < 2) {
        std::cout << "not enough feasible data points to compute a correlation on this machine\n";
        return 0;
    }
    const double rho = spearman_rank_correlation(work_units, seconds);
    std::cout << "Spearman rank correlation (work_units, measured_seconds) = " << rho << "\n";
    std::cout << "This is machine-specific empirical evidence the cost model ranks candidates "
                 "usefully here; it is not a universal wall-clock-time guarantee.\n";
    return 0;
}
