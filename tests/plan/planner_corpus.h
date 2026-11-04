// The fixed planner corpus for Phase 21 (docs/adr/0025 section 5).
//
// One list of plan_job inputs, shared by the planner benchmark
// (benchmarks/planner_benchmark.cpp) and the golden explain() test
// (tests/plan/planner_golden_test.cpp), so that the inputs a speed-up is
// measured on are the same inputs on which it is shown not to change any
// plan.
//
// Every input is deterministic and machine independent:
//   - Pseudo-random choices come from SplitMix64 with a fixed seed per job,
//     never from <random>: the standard distributions are implementation
//     defined, so they would give different circuits on libstdc++ and the MSVC
//     STL.
//   - Every job carries an explicit memory budget, and its HardwareProfile is
//     built by hand. Nothing reads the machine, so the plans (and the golden
//     files) do not depend on how much RAM the machine has or how fast it is.
//
// Size classes follow the step-1 request: small, medium and large, per
// candidate family.

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "menos/accuracy_policy.h"
#include "menos/circuit.h"
#include "menos/noise_method.h"
#include "menos/observable.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"

namespace menos::planner_corpus {

struct CorpusJob {
    std::string name;  // stable: used as the golden file name and benchmark name
    plan::JobSpec spec;
    plan::HardwareProfile hardware;
};

inline constexpr std::uint64_t kBudgetBytes = std::uint64_t{8} << 30;        // 8 GiB
inline constexpr std::uint64_t kTotalMemoryBytes = std::uint64_t{16} << 30;  // 16 GiB

// SplitMix64 (Steele, Lea and Flood 2014): fully specified, so every platform
// produces the same sequence.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // [0, bound)
    std::uint32_t below(std::uint32_t bound) {
        return static_cast<std::uint32_t>(next() % bound);
    }
    // An angle in [0, 2*pi), from the top 53 bits.
    double angle() {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0) * 6.283185307179586;
    }

private:
    std::uint64_t state_;
};

inline plan::HardwareProfile uncalibrated_profile() {
    return plan::HardwareProfile{.total_memory_bytes = kTotalMemoryBytes};
}

// Fixed, hand-written rates standing in for a calibration, for the jobs that
// carry a time budget. Round numbers, not this machine's: a budgeted plan's
// explain() prints est_seconds, which must not depend on the machine.
inline plan::HardwareProfile fixed_calibrated_profile() {
    plan::HardwareProfile profile{.total_memory_bytes = kTotalMemoryBytes};
    profile.work_units_per_second = 1.0e9;
    profile.memory_bandwidth_bytes_per_second = 1.0e10;
    return profile;
}

inline void measure_all(Circuit& circuit) {
    for (std::uint32_t q = 0; q < circuit.num_qubits(); ++q) {
        circuit.append_measure(q, q);
    }
}

// Brickwork: a random rotation on every qubit, then CX on alternating
// nearest-neighbour pairs.
inline Circuit brickwork(std::uint32_t n, std::uint32_t depth, std::uint64_t seed, bool measured) {
    SplitMix64 rng(seed);
    Circuit circuit(n, measured ? n : 0);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(rng.below(2) == 0 ? Opcode::RY : Opcode::RZ, {q}, {rng.angle()});
        }
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    if (measured) {
        measure_all(circuit);
    }
    return circuit;
}

// Random two-qubit pairs at any distance: long spans stress the Schmidt-rank
// bound, which loops over every cut a gate spans.
inline Circuit random_pairs(std::uint32_t n,
                            std::uint32_t depth,
                            std::uint64_t seed,
                            bool measured) {
    SplitMix64 rng(seed);
    Circuit circuit(n, measured ? n : 0);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RX, {q}, {rng.angle()});
        }
        for (std::uint32_t g = 0; g < n / 2; ++g) {
            const std::uint32_t a = rng.below(n);
            std::uint32_t b = rng.below(n - 1);
            if (b >= a) {
                ++b;
            }
            circuit.append(rng.below(2) == 0 ? Opcode::CX : Opcode::CZ, {a, b});
        }
    }
    if (measured) {
        measure_all(circuit);
    }
    return circuit;
}

inline Circuit random_clifford(std::uint32_t n,
                               std::uint32_t depth,
                               std::uint64_t seed,
                               bool measured) {
    SplitMix64 rng(seed);
    Circuit circuit(n, measured ? n : 0);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(rng.below(2) == 0 ? Opcode::H : Opcode::S, {q});
        }
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    if (measured) {
        measure_all(circuit);
    }
    return circuit;
}

// `blocks` independent brickwork blocks of `width` qubits each: the
// interaction graph has `blocks` components, so Factorized is enumerated.
inline Circuit blocks_circuit(std::uint32_t blocks,
                              std::uint32_t width,
                              std::uint32_t depth,
                              std::uint64_t seed,
                              bool measured) {
    SplitMix64 rng(seed);
    const std::uint32_t n = blocks * width;
    Circuit circuit(n, measured ? n : 0);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RY, {q}, {rng.angle()});
        }
        for (std::uint32_t b = 0; b < blocks; ++b) {
            for (std::uint32_t i = layer % 2; i + 1 < width; i += 2) {
                circuit.append(Opcode::CX, {b * width + i, b * width + i + 1});
            }
        }
    }
    if (measured) {
        measure_all(circuit);
    }
    return circuit;
}

// Two brickwork blocks joined by a single CX in the middle of the circuit:
// a one-wire cut separates them, so CutAndRecombine is enumerated.
inline Circuit cut_friendly(std::uint32_t width, std::uint32_t depth, std::uint64_t seed) {
    SplitMix64 rng(seed);
    const std::uint32_t n = 2 * width;
    Circuit circuit(n, 0);
    const auto half = [&](std::uint32_t layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RY, {q}, {rng.angle()});
        }
        for (std::uint32_t b = 0; b < 2; ++b) {
            for (std::uint32_t i = layer % 2; i + 1 < width; i += 2) {
                circuit.append(Opcode::CX, {b * width + i, b * width + i + 1});
            }
        }
    };
    for (std::uint32_t layer = 0; layer < depth / 2; ++layer) {
        half(layer);
    }
    circuit.append(Opcode::CX, {width - 1, width});
    for (std::uint32_t layer = depth / 2; layer < depth; ++layer) {
        half(layer);
    }
    return circuit;
}

// Brickwork with a single-qubit depolarizing channel after every CX target.
inline Circuit noisy_brickwork(std::uint32_t n, std::uint32_t depth, std::uint64_t seed, double p) {
    SplitMix64 rng(seed);
    Circuit circuit(n, n);
    const double a = std::sqrt(1.0 - 3.0 * p / 4.0);
    const double b = std::sqrt(p / 4.0);
    using C = std::complex<double>;
    const std::vector<std::vector<C>> depolarizing{
        {a, 0.0, 0.0, a},                   // sqrt(1-3p/4) I
        {0.0, b, b, 0.0},                   // sqrt(p/4) X
        {0.0, C(0.0, -b), C(0.0, b), 0.0},  // sqrt(p/4) Y
        {b, 0.0, 0.0, -b},                  // sqrt(p/4) Z
    };
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RY, {q}, {rng.angle()});
        }
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
            circuit.append_channel({q + 1}, depolarizing);
        }
    }
    measure_all(circuit);
    return circuit;
}

// --- observables -----------------------------------------------------------

inline PauliString identity_string(std::uint32_t n) {
    const std::size_t words = (n + 63) / 64;
    return PauliString{
        n, std::vector<std::uint64_t>(words, 0), std::vector<std::uint64_t>(words, 0)};
}

// letter: 'X', 'Y' or 'Z'.
inline void set_letter(PauliString& p, std::uint32_t q, char letter) {
    const std::uint64_t bit = std::uint64_t{1} << (q % 64);
    if (letter == 'X' || letter == 'Y') {
        p.x_words[q / 64] |= bit;
    }
    if (letter == 'Z' || letter == 'Y') {
        p.z_words[q / 64] |= bit;
    }
}

inline Observable z_on_every_qubit(std::uint32_t n) {
    PauliString p = identity_string(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        set_letter(p, q, 'Z');
    }
    return Observable{{{p, {1.0, 0.0}}}};
}

// Transverse-field Ising on a chain: Z_i Z_{i+1} for every edge, X_i for every
// qubit. 2n - 1 terms, every support distinct.
inline Observable ising_chain(std::uint32_t n) {
    Observable obs;
    for (std::uint32_t q = 0; q + 1 < n; ++q) {
        PauliString p = identity_string(n);
        set_letter(p, q, 'Z');
        set_letter(p, q + 1, 'Z');
        obs.terms.push_back({p, {1.0, 0.0}});
    }
    for (std::uint32_t q = 0; q < n; ++q) {
        PauliString p = identity_string(n);
        set_letter(p, q, 'X');
        obs.terms.push_back({p, {0.5, 0.0}});
    }
    return obs;
}

// Heisenberg chain: X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} per edge.
// 3(n - 1) terms over n - 1 distinct supports.
inline Observable heisenberg_chain(std::uint32_t n) {
    Observable obs;
    for (std::uint32_t q = 0; q + 1 < n; ++q) {
        for (const char letter : {'X', 'Y', 'Z'}) {
            PauliString p = identity_string(n);
            set_letter(p, q, letter);
            set_letter(p, q + 1, letter);
            obs.terms.push_back({p, {1.0, 0.0}});
        }
    }
    return obs;
}

// `terms` random Pauli strings of weight 2 to 4 (qubits may repeat, so the
// weight can be lower), as in a molecular Hamiltonian of that many terms.
inline Observable random_paulis(std::uint32_t n, std::uint32_t terms, std::uint64_t seed) {
    SplitMix64 rng(seed);
    Observable obs;
    for (std::uint32_t t = 0; t < terms; ++t) {
        PauliString p = identity_string(n);
        const std::uint32_t weight = 2 + rng.below(3);
        for (std::uint32_t w = 0; w < weight; ++w) {
            set_letter(p, rng.below(n), "XYZ"[rng.below(3)]);
        }
        obs.terms.push_back({p, {1.0 / static_cast<double>(t + 1), 0.0}});
    }
    return obs;
}

// --- jobs -------------------------------------------------------------------

inline plan::JobSpec statevector_job(Circuit circuit) {
    return plan::JobSpec{.circuit = std::move(circuit),
                         .shots = 0,
                         .want_statevector = true,
                         .seed = 1,
                         .memory_budget_bytes = kBudgetBytes,
                         .expectation = std::nullopt,
                         .max_threads = 1};
}

inline plan::JobSpec counts_job(Circuit circuit) {
    return plan::JobSpec{.circuit = std::move(circuit),
                         .shots = 1000,
                         .want_statevector = false,
                         .seed = 1,
                         .memory_budget_bytes = kBudgetBytes,
                         .expectation = std::nullopt,
                         .max_threads = 1};
}

inline plan::JobSpec expectation_job(Circuit circuit, Observable observable) {
    return plan::JobSpec{.circuit = std::move(circuit),
                         .shots = 0,
                         .want_statevector = false,
                         .seed = 1,
                         .memory_budget_bytes = kBudgetBytes,
                         .expectation = std::move(observable),
                         .max_threads = 1};
}

// The whole corpus, in a fixed order. Names are "<family>/<size>".
inline std::vector<CorpusJob> planner_corpus() {
    std::vector<CorpusJob> jobs;
    const auto add = [&](std::string name, plan::JobSpec spec, plan::HardwareProfile hw) {
        jobs.push_back(CorpusJob{std::move(name), std::move(spec), hw});
    };
    const plan::HardwareProfile hw = uncalibrated_profile();

    struct Size {
        const char* label;
        std::uint32_t n;
        std::uint32_t depth;
    };
    const Size sizes[] = {{"small", 8, 8}, {"medium", 16, 20}, {"large", 24, 40}};

    // Dense-friendly outputs.
    for (const Size& s : sizes) {
        add(std::string("statevector_brickwork/") + s.label,
            statevector_job(brickwork(s.n, s.depth, 101, false)),
            hw);
    }
    for (const Size& s : sizes) {
        add(std::string("counts_brickwork/") + s.label,
            counts_job(brickwork(s.n, s.depth, 102, true)),
            hw);
    }

    // Expectation jobs: the tensor-network and Pauli candidates are enumerated.
    for (const Size& s : sizes) {
        add(std::string("expectation_z_all/") + s.label,
            expectation_job(brickwork(s.n, s.depth, 103, false), z_on_every_qubit(s.n)),
            hw);
    }
    for (const Size& s : sizes) {
        add(std::string("expectation_ising/") + s.label,
            expectation_job(brickwork(s.n, s.depth, 104, false), ising_chain(s.n)),
            hw);
    }
    for (const Size& s : sizes) {
        add(std::string("expectation_heisenberg/") + s.label,
            expectation_job(brickwork(s.n, s.depth, 105, false), heisenberg_chain(s.n)),
            hw);
    }
    {
        const std::uint32_t terms[] = {20, 100, 400};
        for (std::size_t i = 0; i < 3; ++i) {
            const Size& s = sizes[i];
            add(std::string("expectation_random_paulis/") + s.label,
                expectation_job(brickwork(s.n, s.depth, 106, false),
                                random_paulis(s.n, terms[i], 107)),
                hw);
        }
    }

    // Time-budgeted expectation: branch and bound is off, so every candidate
    // is costed in full.
    for (const Size& s : sizes) {
        plan::JobSpec spec = expectation_job(brickwork(s.n, s.depth, 108, false), ising_chain(s.n));
        spec.time_budget_seconds = 1.0e6;
        add(std::string("expectation_ising_budgeted/") + s.label,
            std::move(spec),
            fixed_calibrated_profile());
    }

    // Clifford: the stabilizer engine wins, at sizes far beyond dense.
    {
        const Size cliff[] = {{"small", 16, 10}, {"medium", 64, 40}, {"large", 256, 100}};
        for (const Size& s : cliff) {
            add(std::string("counts_clifford/") + s.label,
                counts_job(random_clifford(s.n, s.depth, 109, true)),
                hw);
        }
    }

    // Long-range two-qubit gates, beyond dense at the larger sizes: the MPS
    // Schmidt-rank bound is computed for every gate and every cut it spans.
    {
        const Size pairs[] = {{"small", 12, 8}, {"medium", 32, 20}, {"large", 64, 40}};
        for (const Size& s : pairs) {
            add(std::string("counts_random_pairs/") + s.label,
                counts_job(random_pairs(s.n, s.depth, 110, true)),
                hw);
        }
        for (const Size& s : pairs) {
            plan::JobSpec spec =
                expectation_job(random_pairs(s.n, s.depth, 111, false), ising_chain(s.n));
            spec.accuracy_policy = AccuracyPolicy{.kind = AccuracyPolicyKind::Controlled,
                                                  .tolerance = 1e-6,
                                                  .max_bond_dimension = 64};
            add(std::string("expectation_random_pairs_controlled/") + s.label, std::move(spec), hw);
        }
    }

    // Factorized: several independent blocks.
    {
        struct Blocks {
            const char* label;
            std::uint32_t blocks;
            std::uint32_t width;
            std::uint32_t depth;
        };
        const Blocks blocks[] = {{"small", 2, 4, 8}, {"medium", 4, 6, 20}, {"large", 8, 8, 30}};
        for (const Blocks& b : blocks) {
            add(std::string("counts_factorized/") + b.label,
                counts_job(blocks_circuit(b.blocks, b.width, b.depth, 112, true)),
                hw);
        }
        for (const Blocks& b : blocks) {
            add(std::string("expectation_factorized/") + b.label,
                expectation_job(blocks_circuit(b.blocks, b.width, b.depth, 113, false),
                                ising_chain(b.blocks * b.width)),
                hw);
        }
    }

    // CutAndRecombine: two blocks joined by one wire.
    {
        const Size cut[] = {{"small", 4, 6}, {"medium", 8, 12}, {"large", 14, 20}};
        for (const Size& s : cut) {
            add(std::string("expectation_cut/") + s.label,
                expectation_job(cut_friendly(s.n, s.depth, 114), z_on_every_qubit(2 * s.n)),
                hw);
        }
    }

    // Noise: density matrix (default) and trajectories (opted in).
    {
        const Size dm[] = {{"small", 4, 6}, {"medium", 8, 12}, {"large", 12, 20}};
        for (const Size& s : dm) {
            add(std::string("counts_noisy_density_matrix/") + s.label,
                counts_job(noisy_brickwork(s.n, s.depth, 115, 0.01)),
                hw);
        }
        for (const Size& s : dm) {
            plan::JobSpec spec = counts_job(noisy_brickwork(s.n + 4, s.depth, 116, 0.01));
            spec.noise_method = NoiseMethod::Auto;
            add(std::string("counts_noisy_trajectory/") + s.label, std::move(spec), hw);
        }
    }

    return jobs;
}

}  // namespace menos::planner_corpus
