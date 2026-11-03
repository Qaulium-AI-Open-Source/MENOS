// Randomised plan_job inputs for the Phase 21 differential test
// (docs/adr/0025 section 6): the live planner and the frozen d89dde3 planner
// must give the same outcome for every one of them.
//
// Each job is a pure function of its seed, drawn from SplitMix64 like the
// fixed corpus (planner_corpus.h), so a failing seed reproduces on every
// machine. The draws cover what the planner branches on:
//   - circuit structure: generic gate soups (every named gate, CCX, diagonal
//     unitaries, mid-circuit measurement), Clifford, independent blocks
//     (Factorized), two blocks joined by one wire (CutAndRecombine), noise,
//     and wide brickwork or long-range circuits for the tensor network;
//   - output: statevector, counts, expectation (random Pauli sums);
//   - accuracy policy, memory budget, time budget with a calibrated profile,
//     noise method, expectation precision, thread cap, diagonal fusion and a
//     forced CPU backend.
//
// Sizes stay small enough that a Debug ASan build plans each job in well
// under a second. CUDA is never present and never forced: the frozen planner
// is built without CUDA, so a CUDA-capable job would compare two different
// builds rather than two planners.

#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "menos/accuracy_policy.h"
#include "menos/circuit.h"
#include "menos/noise_method.h"
#include "menos/observable.h"
#include "menos/plan.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "planner_corpus.h"

namespace menos::planner_corpus {

namespace random_detail {

inline bool coin(SplitMix64& rng, std::uint32_t one_in) {
    return rng.below(one_in) == 0;
}

// Two distinct qubits of n >= 2.
inline std::pair<QubitId, QubitId> two_qubits(SplitMix64& rng, std::uint32_t n) {
    const std::uint32_t a = rng.below(n);
    std::uint32_t b = rng.below(n - 1);
    if (b >= a) {
        ++b;
    }
    return {a, b};
}

// A diagonal unitary of random phases on k distinct qubits (row-major).
inline void append_diagonal_unitary(Circuit& circuit, SplitMix64& rng, std::uint32_t k) {
    const std::uint32_t n = circuit.num_qubits();
    std::vector<QubitId> qubits;
    while (qubits.size() < k) {
        const QubitId q = rng.below(n);
        bool fresh = true;
        for (const QubitId existing : qubits) {
            fresh = fresh && existing != q;
        }
        if (fresh) {
            qubits.push_back(q);
        }
    }
    const std::size_t dim = std::size_t{1} << k;
    std::vector<std::complex<double>> matrix(dim * dim, {0.0, 0.0});
    for (std::size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] = std::polar(1.0, rng.angle());
    }
    circuit.append_unitary(std::move(qubits), std::move(matrix));
}

// Every named gate, CCX, diagonal unitaries and (if clbits exist) mid-circuit
// measurements, on random qubits.
inline Circuit gate_soup(std::uint32_t n, std::uint32_t depth, SplitMix64& rng, bool measured) {
    static constexpr std::array<Opcode, 15> kOneQubit{Opcode::X,
                                                      Opcode::Y,
                                                      Opcode::Z,
                                                      Opcode::H,
                                                      Opcode::S,
                                                      Opcode::Sdg,
                                                      Opcode::T,
                                                      Opcode::Tdg,
                                                      Opcode::SX,
                                                      Opcode::SXdg,
                                                      Opcode::RX,
                                                      Opcode::RY,
                                                      Opcode::RZ,
                                                      Opcode::P,
                                                      Opcode::U};
    static constexpr std::array<Opcode, 5> kTwoQubit{
        Opcode::CX, Opcode::CZ, Opcode::SWAP, Opcode::CP, Opcode::RZZ};
    Circuit circuit(n, measured ? n : 0);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        const std::uint32_t ops = 1 + rng.below(n + 1);
        for (std::uint32_t o = 0; o < ops; ++o) {
            const std::uint32_t kind = rng.below(20);
            if (kind < 9 || n == 1) {
                const Opcode op = kOneQubit[rng.below(kOneQubit.size())];
                std::vector<double> params;
                if (op == Opcode::RX || op == Opcode::RY || op == Opcode::RZ || op == Opcode::P) {
                    params = {rng.angle()};
                } else if (op == Opcode::U) {
                    params = {rng.angle(), rng.angle(), rng.angle()};
                }
                circuit.append(op, {rng.below(n)}, std::move(params));
            } else if (kind < 17) {
                const Opcode op = kTwoQubit[rng.below(kTwoQubit.size())];
                const auto [a, b] = two_qubits(rng, n);
                std::vector<double> params;
                if (op == Opcode::CP || op == Opcode::RZZ) {
                    params = {rng.angle()};
                }
                circuit.append(op, {a, b}, std::move(params));
            } else if (kind == 17 && n >= 3) {
                const auto [a, b] = two_qubits(rng, n);
                std::uint32_t c = rng.below(n);
                while (c == a || c == b) {
                    c = rng.below(n);
                }
                circuit.append(Opcode::CCX, {a, b, c});
            } else if (kind == 18) {
                append_diagonal_unitary(circuit, rng, n >= 2 && coin(rng, 2) ? 2 : 1);
            } else if (measured) {
                const QubitId q = rng.below(n);
                circuit.append_measure(q, q);
            }
        }
    }
    if (measured) {
        measure_all(circuit);
    }
    return circuit;
}

// A Pauli sum of 1 to 24 terms of weight 0 to 4 (letters may land on the same
// qubit), with occasional duplicate terms.
inline Observable random_observable(std::uint32_t n, SplitMix64& rng) {
    Observable obs;
    const std::uint32_t terms = 1 + rng.below(24);
    for (std::uint32_t t = 0; t < terms; ++t) {
        if (!obs.terms.empty() && coin(rng, 10)) {
            obs.terms.push_back(obs.terms[rng.below(static_cast<std::uint32_t>(obs.terms.size()))]);
            continue;
        }
        PauliString p = identity_string(n);
        const std::uint32_t weight = rng.below(5);
        for (std::uint32_t w = 0; w < weight; ++w) {
            set_letter(p, rng.below(n), "XYZ"[rng.below(3)]);
        }
        const double re = static_cast<double>(1 + rng.below(16)) / 8.0;
        obs.terms.push_back({p, {coin(rng, 2) ? re : -re, 0.0}});
    }
    return obs;
}

// A power of two in [2^lo, 2^hi].
inline std::uint64_t power_of_two(SplitMix64& rng, std::uint32_t lo, std::uint32_t hi) {
    return std::uint64_t{1} << (lo + rng.below(hi - lo + 1));
}

}  // namespace random_detail

// The job for one seed. The name records the seed and the drawn family, so a
// failure message says what was being planned.
inline CorpusJob random_job(std::uint64_t seed) {
    using namespace random_detail;
    SplitMix64 rng(seed);
    const std::uint32_t family = rng.below(7);
    // 0 statevector, 1 counts, 2 expectation. The "wide" family is always an
    // expectation: it exists to give the tensor-network costing (greedy
    // min-fill, the Phase 21 hot spot) graphs of realistic size.
    const std::uint32_t output = family == 6 ? 2 : rng.below(3);
    const bool measured = output == 1;
    const std::uint64_t circuit_seed = rng.next();

    std::string family_name;
    Circuit circuit(1, 0);
    switch (family) {
        case 0: {
            family_name = "soup";
            const std::uint32_t n = 1 + rng.below(18);
            circuit = gate_soup(n, rng.below(14), rng, measured);
            break;
        }
        case 1: {
            family_name = "clifford";
            circuit = random_clifford(2 + rng.below(80), rng.below(30), circuit_seed, measured);
            break;
        }
        case 2: {
            family_name = "blocks";
            circuit = blocks_circuit(
                2 + rng.below(4), 1 + rng.below(6), rng.below(16), circuit_seed, measured);
            break;
        }
        case 3: {
            family_name = "cut";
            circuit = cut_friendly(2 + rng.below(8), 2 + rng.below(12), circuit_seed);
            if (measured) {
                Circuit with_clbits(circuit.num_qubits(), circuit.num_qubits());
                for (const Operation& op : circuit.operations()) {
                    const auto& gate = std::get<GateOp>(op);
                    with_clbits.append(gate.opcode, gate.qubits, gate.params);
                }
                measure_all(with_clbits);
                circuit = std::move(with_clbits);
            }
            break;
        }
        case 4: {
            family_name = "pairs";
            circuit = random_pairs(2 + rng.below(30), rng.below(12), circuit_seed, measured);
            break;
        }
        case 6: {
            family_name = "wide";
            const std::uint32_t n = 16 + rng.below(25);
            const std::uint32_t depth = 2 + rng.below(19);
            circuit = coin(rng, 2) ? brickwork(n, depth, circuit_seed, false)
                                   : random_pairs(n, depth / 2, circuit_seed, false);
            break;
        }
        default: {
            family_name = "noisy";
            const double p = static_cast<double>(1 + rng.below(50)) / 1000.0;
            circuit = noisy_brickwork(2 + rng.below(8), rng.below(10), circuit_seed, p);
            break;
        }
    }

    const std::uint32_t n = circuit.num_qubits();
    plan::JobSpec spec = output == 0 ? statevector_job(std::move(circuit))
                         : output == 1
                             ? counts_job(std::move(circuit))
                             : expectation_job(std::move(circuit), random_observable(n, rng));
    if (output == 1) {
        spec.shots = 1 + rng.below(4096);
    }
    spec.seed = rng.next();
    spec.memory_budget_bytes =
        coin(rng, 4) ? std::nullopt : std::optional<std::uint64_t>(power_of_two(rng, 10, 34));
    if (coin(rng, 3)) {
        spec.accuracy_policy.kind = AccuracyPolicyKind::Controlled;
        spec.accuracy_policy.tolerance = coin(rng, 2) ? 1e-6 : 1e-10;
        if (coin(rng, 2)) {
            spec.accuracy_policy.max_bond_dimension = 1U << (1 + rng.below(8));
        }
    }
    if (coin(rng, 2)) {
        spec.noise_method = NoiseMethod::Auto;
    }
    if (output == 2 && coin(rng, 3)) {
        spec.expectation_precision = coin(rng, 2) ? 1e-2 : 1e-3;
    }
    spec.max_threads = 1 + rng.below(8);
    spec.fuse_diagonal_gates = coin(rng, 4);
    if (coin(rng, 8)) {
        spec.backend = Backend::Cpu;
    }

    plan::HardwareProfile hardware{.total_memory_bytes = power_of_two(rng, 20, 36)};
    if (coin(rng, 3)) {
        hardware.work_units_per_second = static_cast<double>(1 + rng.below(1000)) * 1.0e7;
        hardware.memory_bandwidth_bytes_per_second =
            static_cast<double>(1 + rng.below(100)) * 1.0e9;
        if (coin(rng, 2)) {
            // From 10 microseconds to about 3 hours.
            spec.time_budget_seconds = 1.0e-5 * static_cast<double>(power_of_two(rng, 0, 30));
        }
    }

    return CorpusJob{"random/" + std::to_string(seed) + "/" + family_name + "/" +
                         (output == 0   ? "statevector"
                          : output == 1 ? "counts"
                                        : "expectation"),
                     std::move(spec),
                     hardware};
}

}  // namespace menos::planner_corpus
