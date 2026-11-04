// Phase 23 step 1, analysis (iv) at the gate sizes the benchmark's table skips:
// the pass counts of docs/adr/0027 section 3.2 for the four families at n = 29
// (the host-offloaded tier's gate size), 31 (the disk tier's) and 33, for every
// L in 25..28 and w in gmax..gmax+2, through the same count_passes and the same
// family builders as benchmarks/tiered_memory_benchmark.cpp (copied verbatim).
// Also the passes of each family's first 100 operations, for the Q7 rule.
// Rows at n = 28 are printed for comparison with the benchmark's own table.
//
// Build (WSL, after build/bench exists):
//   g++ -std=c++20 -O2 -I include -I benchmarks pass_counts_at_gate_sizes.cpp \
//       build/bench/src/ir/libmenos_ir.a build/bench/src/common/libmenos_common.a
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "crossover_circuit.h"
#include "menos/circuit.h"
#include "tiered_pass_count.h"

namespace {

menos::Circuit ladder(std::uint32_t n) {
    menos::Circuit circuit(n, 0);
    for (int layer = 0; layer < menos::benchmarks::kCrossoverLayers; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(menos::Opcode::H, {static_cast<menos::QubitId>(q)});
        }
        for (std::uint32_t q = 0; q + 1 < n; ++q) {
            circuit.append(menos::Opcode::CX,
                           {static_cast<menos::QubitId>(q), static_cast<menos::QubitId>(q + 1)});
        }
    }
    return circuit;
}

menos::Circuit random_pairs(std::uint32_t n) {
    constexpr int kGates = 200;
    std::mt19937_64 rng(1);
    std::uniform_int_distribution<std::uint32_t> pick(0, n - 1);
    menos::Circuit circuit(n, 0);
    for (int g = 0; g < kGates; ++g) {
        const std::uint32_t a = pick(rng);
        std::uint32_t b = pick(rng);
        while (b == a) {
            b = pick(rng);
        }
        if (g % 2 == 0) {
            circuit.append(menos::Opcode::RX, {static_cast<menos::QubitId>(a)}, {0.3});
        } else {
            circuit.append(menos::Opcode::CX,
                           {static_cast<menos::QubitId>(a), static_cast<menos::QubitId>(b)});
        }
    }
    return circuit;
}

menos::Circuit qft_like(std::uint32_t n) {
    menos::Circuit circuit(n, 0);
    for (std::uint32_t i = 0; i < n; ++i) {
        circuit.append(menos::Opcode::H, {static_cast<menos::QubitId>(i)});
        for (std::uint32_t j = i + 1; j < n; ++j) {
            circuit.append(menos::Opcode::CP,
                           {static_cast<menos::QubitId>(j), static_cast<menos::QubitId>(i)},
                           {3.141592653589793 / static_cast<double>(std::uint64_t{1} << (j - i))});
        }
    }
    return circuit;
}

}  // namespace

int main() {
    using menos::benchmarks::count_passes;
    using menos::benchmarks::PassCountRequest;
    const std::vector<std::pair<std::string, menos::Circuit (*)(std::uint32_t)>> families = {
        {"crossover", menos::benchmarks::make_crossover_circuit},
        {"ladder", ladder},
        {"random_pairs", random_pairs},
        {"qft_like", qft_like},
    };
    // The table goes to stderr, the JSON array to stdout.
    std::fprintf(stderr,
                 "family         n gates   L global gmax  w=gmax  w+1  w+2 | first 100 ops: global "
                 "gmax w=gmax w+1 w+2\n");
    std::printf("[\n");
    bool first = true;
    for (const auto& [name, make] : families) {
        for (const std::uint32_t n : {28U, 29U, 31U, 33U}) {
            const menos::Circuit circuit = make(n);
            const auto& ops = circuit.operations();
            const std::span<const menos::Operation> all(ops.data(), ops.size());
            const std::span<const menos::Operation> prefix(ops.data(),
                                                           std::min<std::size_t>(100, ops.size()));
            for (const std::uint32_t L : {24U, 25U, 26U, 27U, 28U}) {
                if (L >= n)
                    continue;
                const auto gmax = count_passes(all, PassCountRequest{L, 64}).max_global_operands;
                const auto p0 = count_passes(all, PassCountRequest{L, gmax});
                const auto p1 = count_passes(all, PassCountRequest{L, gmax + 1});
                const auto p2 = count_passes(all, PassCountRequest{L, gmax + 2});
                const auto q0 = count_passes(prefix, PassCountRequest{L, gmax});
                const auto q1 = count_passes(prefix, PassCountRequest{L, gmax + 1});
                const auto q2 = count_passes(prefix, PassCountRequest{L, gmax + 2});
                std::fprintf(stderr,
                             "%-12s %3u %5zu  %2u %6llu %4u %7llu %4llu %4llu | %6llu %4u %6llu "
                             "%4llu %4llu\n",
                             name.c_str(),
                             n,
                             ops.size(),
                             L,
                             static_cast<unsigned long long>(p0.global_gates),
                             gmax,
                             static_cast<unsigned long long>(p0.passes),
                             static_cast<unsigned long long>(p1.passes),
                             static_cast<unsigned long long>(p2.passes),
                             static_cast<unsigned long long>(q0.global_gates),
                             q0.max_global_operands,
                             static_cast<unsigned long long>(q0.passes),
                             static_cast<unsigned long long>(q1.passes),
                             static_cast<unsigned long long>(q2.passes));
                std::printf(
                    "%s{\"family\":\"%s\",\"n\":%u,\"gates\":%zu,\"local_qubits\":%u,"
                    "\"global_gates\":%llu,\"max_global_operands\":%u,"
                    "\"passes_window_gmax\":%llu,\"passes_window_gmax_plus_1\":%llu,"
                    "\"passes_window_gmax_plus_2\":%llu,"
                    "\"first_100_ops\":{\"global_gates\":%llu,\"max_global_operands\":%u,"
                    "\"passes_window_gmax\":%llu,\"passes_window_gmax_plus_1\":%llu,"
                    "\"passes_window_gmax_plus_2\":%llu}}\n",
                    first ? " " : ",",
                    name.c_str(),
                    n,
                    ops.size(),
                    L,
                    static_cast<unsigned long long>(p0.global_gates),
                    gmax,
                    static_cast<unsigned long long>(p0.passes),
                    static_cast<unsigned long long>(p1.passes),
                    static_cast<unsigned long long>(p2.passes),
                    static_cast<unsigned long long>(q0.global_gates),
                    q0.max_global_operands,
                    static_cast<unsigned long long>(q0.passes),
                    static_cast<unsigned long long>(q1.passes),
                    static_cast<unsigned long long>(q2.passes));
                first = false;
            }
        }
    }
    std::printf("]\n");
    return 0;
}
