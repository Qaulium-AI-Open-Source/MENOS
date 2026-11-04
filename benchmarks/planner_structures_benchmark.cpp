// Microbenchmarks for the planner data structures of Phase 21 step 2
// (docs/adr/0025 section 12), each timed alone so a change to one can be
// measured without the noise of a whole plan_job:
//   fill/<n>/<density%>    compute_fill over every vertex of a random graph
//                          (bit_adjacency.h, change (d))
//   min_fill/<circuit>     the dense min-fill search on the closed network's
//                          line graph of a corpus-style circuit, Z on every
//                          qubit (min_fill_queue.h and bit_adjacency.h,
//                          changes (b) and (d))
//   rank/<gate>/<case>     one operator-Schmidt-rank lookup: an SVD alone,
//                          a cache miss, a cache hit (schmidt_rank_internal.h,
//                          change (c))
//
// Run (release build with -DMENOS_BUILD_BENCHMARKS=ON):
//   ./menos_planner_structures_benchmark --benchmark_repetitions=5

#include <benchmark/benchmark.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "analysis/bit_adjacency.h"
#include "analysis/schmidt_rank_internal.h"
#include "analysis/treewidth.h"
#include "ir/gate_table.h"
#include "planner_corpus.h"

namespace {

using menos::analysis::detail::BitAdjacency;

struct RandomGraph {
    std::size_t n;
    double density;
};

BitAdjacency random_adjacency(RandomGraph shape) {
    const std::size_t n = shape.n;
    BitAdjacency adjacency(n);
    std::mt19937_64 rng(20261104);
    std::bernoulli_distribution edge(shape.density);
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a + 1; b < n; ++b) {
            if (edge(rng)) {
                adjacency.set(a, b);
                adjacency.set(b, a);
            }
        }
    }
    return adjacency;
}

void fill_benchmark(benchmark::State& state, RandomGraph shape) {
    const std::size_t n = shape.n;
    const BitAdjacency adjacency = random_adjacency(shape);
    for (auto _ : state) {
        std::size_t total = 0;
        for (std::size_t v = 0; v < n; ++v) {
            total += menos::analysis::detail::compute_fill(adjacency, v);
        }
        benchmark::DoNotOptimize(total);
    }
}

menos::analysis::TensorNetworkGraph line_graph_of(const menos::Circuit& circuit) {
    const std::uint32_t n = circuit.num_qubits();
    menos::PauliString z = menos::planner_corpus::identity_string(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        menos::planner_corpus::set_letter(z, q, 'Z');
    }
    return menos::analysis::build_line_graph(
               menos::analysis::build_closed_timelines(n, circuit.operations(), z))
        .graph;
}

void min_fill_benchmark(benchmark::State& state, const menos::Circuit& circuit) {
    const menos::analysis::TensorNetworkGraph graph = line_graph_of(circuit);
    for (auto _ : state) {
        menos::analysis::EliminationOrder order =
            menos::analysis::estimate_treewidth_min_fill_dense(graph);
        benchmark::DoNotOptimize(order);
    }
    state.counters["nodes"] = static_cast<double>(graph.num_nodes);
    state.counters["edges"] = static_cast<double>(graph.edges.size());
}

enum class RankCase { Svd, Miss, Hit };

void rank_benchmark(benchmark::State& state,
                    const std::vector<menos::analysis::detail::Complex>& u,
                    std::size_t k,
                    RankCase which) {
    const std::vector<std::size_t> left{0};
    std::vector<std::size_t> right;
    for (std::size_t p = 1; p < k; ++p) {
        right.push_back(p);
    }
    menos::analysis::detail::SchmidtRankCache warm;
    static_cast<void>(warm.rank(u, k, left, right));
    for (auto _ : state) {
        if (which == RankCase::Svd) {
            benchmark::DoNotOptimize(
                menos::analysis::detail::operator_schmidt_rank(u, k, left, right));
        } else if (which == RankCase::Miss) {
            menos::analysis::detail::SchmidtRankCache cold;
            benchmark::DoNotOptimize(cold.rank(u, k, left, right));
        } else {
            benchmark::DoNotOptimize(warm.rank(u, k, left, right));
        }
    }
}

std::vector<std::complex<double>> gate_matrix(menos::Opcode op) {
    return menos::ir::generate_matrix(op, std::vector<double>{});
}

[[maybe_unused]] const bool kRegistered = [] {
    for (const std::size_t n : {256U, 2048U}) {
        for (const int percent : {1, 10, 50}) {
            benchmark::RegisterBenchmark(
                "fill/" + std::to_string(n) + "/" + std::to_string(percent),
                [n, percent](benchmark::State& s) {
                    fill_benchmark(s, {n, static_cast<double>(percent) / 100.0});
                });
        }
    }
    using menos::planner_corpus::brickwork;
    using menos::planner_corpus::random_pairs;
    const std::vector<std::pair<std::string, menos::Circuit>> circuits{
        {"brickwork/16x8", brickwork(16, 8, 1, false)},
        {"brickwork/24x12", brickwork(24, 12, 2, false)},
        {"brickwork/32x16", brickwork(32, 16, 3, false)},
        {"random_pairs/24x6", random_pairs(24, 6, 4, false)},
        {"random_pairs/40x8", random_pairs(40, 8, 5, false)},
    };
    for (const auto& [name, circuit] : circuits) {
        benchmark::RegisterBenchmark("min_fill/" + name, [circuit](benchmark::State& s) {
            min_fill_benchmark(s, circuit);
        })->Unit(benchmark::kMicrosecond);
    }
    const std::vector<std::tuple<std::string, menos::Opcode, std::size_t>> gates{
        {"cx", menos::Opcode::CX, 2},
        {"swap", menos::Opcode::SWAP, 2},
        {"ccx", menos::Opcode::CCX, 3}};
    for (const auto& [name, op, k] : gates) {
        const std::vector<std::complex<double>> u = gate_matrix(op);
        benchmark::RegisterBenchmark("rank/" + name + "/svd", [u, k](benchmark::State& s) {
            rank_benchmark(s, u, k, RankCase::Svd);
        });
        benchmark::RegisterBenchmark("rank/" + name + "/miss", [u, k](benchmark::State& s) {
            rank_benchmark(s, u, k, RankCase::Miss);
        });
        benchmark::RegisterBenchmark("rank/" + name + "/hit", [u, k](benchmark::State& s) {
            rank_benchmark(s, u, k, RankCase::Hit);
        });
    }
    return true;
}();

}  // namespace
