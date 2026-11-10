#include "plan/time_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/simulation_engine.h"
#include "plan/available_engines.h"

namespace menos::plan {

namespace {

using Clock = std::chrono::steady_clock;
using Counts = std::array<double, kTimeComponents>;

// Each calibration run is repeated until it has taken this long, at least
// kMinRepetitions times, after one discarded run (warm caches, page faults
// and clocks), the policy of the dense rate's calibration
// (hardware_profile.cpp).
constexpr double kDeadlineSeconds = 0.01;
constexpr int kMinRepetitions = 3;

// One calibration measurement: a job for one engine, built so that one of the
// engine's time components dominates it.
struct Case {
    Circuit circuit;
    bool want_statevector = false;
    bool want_counts = false;
    std::uint64_t shots = 0;
    std::optional<Observable> observable = std::nullopt;
    std::uint64_t trajectories = 0;
};

// The circuit and observable builders below take a register size beside a
// qubit, gate, layer or term count; each is named at its one call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
PauliString pauli_z(std::uint32_t n, std::uint32_t qubit) {
    const std::size_t words = (static_cast<std::size_t>(n) + 63) / 64;
    PauliString p{n, std::vector<std::uint64_t>(words, 0), std::vector<std::uint64_t>(words, 0)};
    p.z_words[qubit / 64] |= std::uint64_t{1} << (qubit % 64);
    return p;
}

Observable z_on(std::uint32_t n, std::uint32_t qubit) {
    return Observable{{{pauli_z(n, qubit), {1.0, 0.0}}}};
}

// t terms, term i a Z on qubit i mod n and an X on qubit (3i + 1) mod n: a
// spread of supports, the same on every platform.
Observable mixed_terms(std::uint32_t n, std::uint32_t t) {
    Observable o;
    for (std::uint32_t i = 0; i < t; ++i) {
        PauliString p = pauli_z(n, i % n);
        const std::uint32_t x = (3 * i + 1) % n;
        p.x_words[x / 64] |= std::uint64_t{1} << (x % 64);
        o.terms.emplace_back(std::move(p), std::complex<double>{1.0, 0.0});
    }
    return o;
}

// SplitMix64: a reproducible bit stream, the same on every platform.
std::uint64_t mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// t Pauli strings with independent random X and Z bits on every qubit: the
// dense supports of a generic observable.
Observable random_terms(std::uint32_t n, std::uint32_t t) {
    const std::size_t words = (static_cast<std::size_t>(n) + 63) / 64;
    Observable o;
    for (std::uint32_t i = 0; i < t; ++i) {
        PauliString p{
            n, std::vector<std::uint64_t>(words, 0), std::vector<std::uint64_t>(words, 0)};
        for (std::size_t w = 0; w < words; ++w) {
            const std::uint32_t bits =
                std::min<std::uint32_t>(64, n - static_cast<std::uint32_t>(64 * w));
            const std::uint64_t mask =
                bits == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
            const std::uint64_t row = 2 * static_cast<std::uint64_t>(i);
            p.x_words[w] = mix((row * words) + w) & mask;
            p.z_words[w] = mix(((row + 1) * words) + w) & mask;
        }
        o.terms.emplace_back(std::move(p), std::complex<double>{1.0, 0.0});
    }
    return o;
}

Observable all_z(std::uint32_t n) {
    Observable o;
    for (std::uint32_t q = 0; q < n; ++q) {
        o.terms.emplace_back(pauli_z(n, q), std::complex<double>{1.0, 0.0});
    }
    return o;
}

// Layers of H on every qubit then CX on neighbouring pairs, alternating the
// pairing: the dense rate calibration's circuit shape.
Circuit h_cx_layers(std::uint32_t n, std::uint32_t layers) {
    Circuit c(n, n);
    for (std::uint32_t l = 0; l < layers; ++l) {
        for (std::uint32_t q = 0; q < n; ++q) {
            c.append(Opcode::H, {q});
        }
        for (std::uint32_t q = l % 2; q + 1 < n; q += 2) {
            c.append(Opcode::CX, {q, q + 1});
        }
    }
    return c;
}

// A Clifford circuit of `gates` gates cycling H, S and CX over the register.
Circuit clifford(std::uint32_t n, std::uint32_t gates) {
    Circuit c(n, n);
    for (std::uint32_t g = 0; g < gates; ++g) {
        const std::uint32_t q = (g * 7) % n;
        switch (g % 3) {
            case 0:
                c.append(Opcode::H, {q});
                break;
            case 1:
                c.append(Opcode::S, {q});
                break;
            default:
                c.append(Opcode::CX, {q, (q + 1) % n});
                break;
        }
    }
    return c;
}

// Rotation layers and neighbouring CX: an entangling circuit whose MPS bonds
// grow with depth.
Circuit ry_cx_layers(std::uint32_t n, std::uint32_t layers) {
    Circuit c(n, n);
    for (std::uint32_t l = 0; l < layers; ++l) {
        for (std::uint32_t q = 0; q < n; ++q) {
            c.append(Opcode::RY, {q}, {0.3 + (0.17 * static_cast<double>(q + l))});
        }
        for (std::uint32_t q = l % 2; q + 1 < n; q += 2) {
            c.append(Opcode::CX, {q, q + 1});
        }
    }
    return c;
}

// Two-qubit gates on the disjoint pairs (2i, 2i + 1) only, with a rotation
// between them: no gate ever crosses the cut between pairs, so every MPS bond
// is at most 2 whatever the depth (the exact policy keeps zero singular
// values, so a bond is bounded by structure, not by entanglement), and the
// per-update cost shows without the chi^3 term.
Circuit paired_gates(std::uint32_t n, std::uint32_t gates) {
    Circuit c(n, n);
    for (std::uint32_t g = 0; g < gates; ++g) {
        const std::uint32_t q = 2 * ((g * 3) % (n / 2));
        c.append(Opcode::RY, {q}, {0.2 + (0.01 * static_cast<double>(g % 50))});
        c.append(Opcode::CX, {q, q + 1});
    }
    return c;
}

// Trotter steps of a transverse-field Ising chain: RZZ on the even then the
// odd bonds, then RX on every qubit -- the shape of a physics workload.
Circuit ising_trotter(std::uint32_t n, std::uint32_t steps) {
    Circuit c(n, n);
    for (std::uint32_t s = 0; s < steps; ++s) {
        for (std::uint32_t parity = 0; parity < 2; ++parity) {
            for (std::uint32_t q = parity; q + 1 < n; q += 2) {
                c.append(Opcode::RZZ, {q, q + 1}, {0.2});
            }
        }
        for (std::uint32_t q = 0; q < n; ++q) {
            c.append(Opcode::RX, {q}, {0.3});
        }
    }
    return c;
}

// circuit followed by `gates` one-qubit rotations spread over the register.
Circuit then_one_qubit_gates(Circuit c, std::uint32_t gates) {
    const std::uint32_t n = c.num_qubits();
    for (std::uint32_t g = 0; g < gates; ++g) {
        c.append(Opcode::RY, {(g * 5) % n}, {0.1 + (0.01 * static_cast<double>(g % 50))});
    }
    return c;
}

// A bit-flip channel of probability p on one qubit.
void bit_flip(Circuit& c, std::uint32_t q, double p) {
    const std::complex<double> a{std::sqrt(1.0 - p), 0.0};
    const std::complex<double> b{std::sqrt(p), 0.0};
    c.append_channel({q}, {{a, {0.0, 0.0}, {0.0, 0.0}, a}, {{0.0, 0.0}, b, b, {0.0, 0.0}}});
}

// h_cx_layers with a bit flip after every layer on every qubit.
Circuit noisy_layers(std::uint32_t n, std::uint32_t layers) {
    Circuit c(n, n);
    for (std::uint32_t l = 0; l < layers; ++l) {
        for (std::uint32_t q = 0; q < n; ++q) {
            c.append(Opcode::H, {q});
        }
        for (std::uint32_t q = l % 2; q + 1 < n; q += 2) {
            c.append(Opcode::CX, {q, q + 1});
        }
        for (std::uint32_t q = 0; q < n; ++q) {
            bit_flip(c, q, 0.01);
        }
    }
    return c;
}

// circuit with an RZ on qubit 0 in front.
Circuit with_rz(const Circuit& circuit) {
    Circuit c(circuit.num_qubits(), circuit.num_clbits());
    c.append(Opcode::RZ, {0}, {0.3});
    for (const Operation& op : circuit.operations()) {
        const auto& gate = std::get<GateOp>(op);
        c.append(gate.opcode, gate.qubits, gate.params);
    }
    return c;
}

// NOLINTEND(bugprone-easily-swappable-parameters)

Case counts_case(Circuit circuit, std::uint64_t shots) {
    return Case{.circuit = std::move(circuit), .want_counts = true, .shots = shots};
}

Case expectation_case(Circuit circuit, Observable observable, std::uint64_t trajectories = 0) {
    return Case{.circuit = std::move(circuit),
                .observable = std::move(observable),
                .trajectories = trajectories};
}

// The calibration cases of each engine: one per component, sized so that
// component dominates and one run takes a few milliseconds.
std::vector<Case> cases_for(Engine engine) {
    std::vector<Case> cases;
    switch (engine) {
        case Engine::Dense:
            cases.push_back(Case{.circuit = h_cx_layers(4, 2000), .want_statevector = true});
            cases.push_back(Case{.circuit = h_cx_layers(14, 4), .want_statevector = true});
            cases.push_back(counts_case(h_cx_layers(4, 1), 200000));
            break;
        case Engine::Stabilizer:
            cases.push_back(counts_case(clifford(8, 30000), 1));
            cases.push_back(counts_case(clifford(512, 6000), 1));
            cases.push_back(counts_case(clifford(512, 512), 20000));
            break;
        case Engine::Pauli:
            // One RZ makes the circuit non-Clifford, which the Pauli engine
            // requires (a Clifford one goes to the stabilizer engine); placed
            // first, on |0>-side terms it splits nothing.
            cases.push_back(expectation_case(with_rz(clifford(8, 30000)), z_on(8, 0)));
            cases.push_back(expectation_case(with_rz(clifford(40, 300)), random_terms(40, 300)));
            cases.push_back(expectation_case(with_rz(clifford(256, 300)), random_terms(256, 300)));
            break;
        case Engine::MPS:
            // Bonds at most 2: the per-update cost. Bonds up to 64: chi^3.
            // Bonds up to 16 with many one-site gates after them: chi^2.
            cases.push_back(counts_case(paired_gates(16, 2000), 1));
            cases.push_back(counts_case(ry_cx_layers(12, 6), 1));
            cases.push_back(counts_case(then_one_qubit_gates(ry_cx_layers(10, 4), 3000), 1));
            cases.push_back(counts_case(ising_trotter(12, 3), 1));
            cases.push_back(counts_case(ry_cx_layers(10, 3), 20000));
            break;
        case Engine::TensorNetwork:
            cases.push_back(expectation_case(h_cx_layers(10, 2), all_z(10)));
            cases.push_back(expectation_case(ry_cx_layers(14, 4), z_on(14, 7)));
            cases.push_back(expectation_case(h_cx_layers(6, 1), mixed_terms(6, 60)));
            break;
        case Engine::DensityMatrix:
            cases.push_back(counts_case(noisy_layers(2, 800), 1));
            cases.push_back(counts_case(noisy_layers(7, 3), 1));
            cases.push_back(counts_case(noisy_layers(2, 1), 100000));
            break;
        case Engine::Trajectory:
            cases.push_back(expectation_case(noisy_layers(2, 10), z_on(2, 0), 3000));
            cases.push_back(expectation_case(noisy_layers(11, 4), z_on(11, 0), 8));
            cases.push_back(expectation_case(noisy_layers(1, 1), z_on(1, 0), 30000));
            break;
    }
    return cases;
}

const SimulationEngine* cpu_engine(Engine id) {
    if (id == Engine::Trajectory) {
        return &trajectory_engine();
    }
    for (const SimulationEngine* engine : noisy_job_engines()) {
        if (engine->engine_id() == id && engine->backend() == Backend::Cpu &&
            engine->tier() == Tier::InMemory) {
            return engine;
        }
    }
    return nullptr;
}

// One measurement: the case's time terms for `engine` and its measured time,
// or nullopt when the engine cannot run it.
std::optional<std::pair<Counts, double>> measure(const SimulationEngine& engine, const Case& c) {
    const std::uint32_t n = c.circuit.num_qubits();
    const Observable* observable = c.observable.has_value() ? &*c.observable : nullptr;
    const CostEstimate estimate = engine.estimate(EngineEstimateRequest{
        .num_qubits = n,
        .operations = &c.circuit.operations(),
        .want_statevector = c.want_statevector,
        .want_counts = c.want_counts,
        .want_expectation = observable != nullptr,
        .observable = observable,
        .trajectories = c.trajectories,
        .shots = c.shots,
        .bitstring_width = c.want_counts ? n : 0,
        .is_sole_region = true,
    });
    if (!estimate.feasible) {
        return std::nullopt;
    }
    std::vector<std::pair<QubitId, ClbitId>> targets;
    if (c.want_counts) {
        for (std::uint32_t q = 0; q < n; ++q) {
            targets.emplace_back(q, q);
        }
    }
    const EngineExecutionRequest request{
        .num_qubits = n,
        .operations = c.circuit.operations(),
        .want_statevector = c.want_statevector,
        .want_counts = c.want_counts,
        .want_expectation = observable != nullptr,
        .shots = c.shots,
        .measure_targets = targets,
        .bitstring_width = c.want_counts ? n : 0,
        .seed = 1,
        .global_phase = 0.0,
        .budget_bytes = estimate.estimated_peak_bytes,
        .is_sole_region = true,
        .observable = observable,
        .trajectories = c.trajectories,
        .max_threads = 1,
    };
    static_cast<void>(engine.execute(request));  // discarded
    int repetitions = 0;
    const Clock::time_point start = Clock::now();
    double seconds = 0.0;
    do {
        static_cast<void>(engine.execute(request));
        ++repetitions;
        seconds = std::chrono::duration<double>(Clock::now() - start).count();
    } while (seconds < kDeadlineSeconds || repetitions < kMinRepetitions);
    const auto& counts = estimate.time.counts[static_cast<std::size_t>(engine.engine_id())];
    return std::pair<Counts, double>{counts, seconds / repetitions};
}

// Solves the k x k system m x = v by Gaussian elimination with partial
// pivoting; nullopt if it is singular.
std::optional<std::vector<double>> solve(std::vector<std::vector<double>> m,
                                         std::vector<double> v) {
    const std::size_t k = v.size();
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t pivot = i;
        for (std::size_t r = i + 1; r < k; ++r) {
            if (std::abs(m[r][i]) > std::abs(m[pivot][i])) {
                pivot = r;
            }
        }
        if (std::abs(m[pivot][i]) < 1e-300) {
            return std::nullopt;
        }
        std::swap(m[i], m[pivot]);
        std::swap(v[i], v[pivot]);
        for (std::size_t r = 0; r < k; ++r) {
            if (r == i) {
                continue;
            }
            const double f = m[r][i] / m[i][i];
            for (std::size_t c = i; c < k; ++c) {
                m[r][c] -= f * m[i][c];
            }
            v[r] -= f * v[i];
        }
    }
    std::vector<double> x(k);
    for (std::size_t i = 0; i < k; ++i) {
        x[i] = v[i] / m[i][i];
    }
    return x;
}

}  // namespace

std::array<double, kTimeComponents> fit_seconds_per_count(const std::vector<Counts>& rows,
                                                          const std::vector<double>& seconds) {
    // Relative error: row i scaled by 1 / seconds[i], target 1. Each column is
    // also scaled to unit maximum so the normal equations stay well
    // conditioned across counts that differ by orders of magnitude.
    const std::size_t n = rows.size();
    std::array<double, kTimeComponents> scale{};
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < kTimeComponents; ++c) {
            scale[c] = std::max(scale[c], rows[i][c] / seconds[i]);
        }
    }
    std::vector<std::size_t> live;
    for (std::size_t c = 0; c < kTimeComponents; ++c) {
        if (scale[c] > 0.0) {
            live.push_back(c);
        }
    }
    // Non-negative least squares by enumerating the column subsets: the
    // best fit with every coefficient non-negative is the least-squares fit
    // on one of them. kTimeComponents columns at most, so at most 15 subsets.
    std::array<double, kTimeComponents> best{};
    double best_residual = -1.0;
    for (std::size_t mask = 1; mask < (std::size_t{1} << live.size()); ++mask) {
        std::vector<std::size_t> cols;
        for (std::size_t j = 0; j < live.size(); ++j) {
            if (((mask >> j) & 1U) != 0) {
                cols.push_back(live[j]);
            }
        }
        std::vector<std::vector<double>> normal(cols.size(), std::vector<double>(cols.size(), 0.0));
        std::vector<double> rhs(cols.size(), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t a = 0; a < cols.size(); ++a) {
                const double xa = rows[i][cols[a]] / seconds[i] / scale[cols[a]];
                rhs[a] += xa;
                for (std::size_t b = 0; b < cols.size(); ++b) {
                    normal[a][b] += xa * (rows[i][cols[b]] / seconds[i] / scale[cols[b]]);
                }
            }
        }
        const std::optional<std::vector<double>> x = solve(normal, rhs);
        if (!x.has_value() || std::any_of(x->begin(), x->end(), [](double v) { return v < 0.0; })) {
            continue;
        }
        std::array<double, kTimeComponents> candidate{};
        for (std::size_t a = 0; a < cols.size(); ++a) {
            candidate[cols[a]] = (*x)[a] / scale[cols[a]];
        }
        double residual = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double predicted = 0.0;
            for (std::size_t c = 0; c < kTimeComponents; ++c) {
                predicted += rows[i][c] * candidate[c];
            }
            const double e = (predicted / seconds[i]) - 1.0;
            residual += e * e;
        }
        if (best_residual < 0.0 || residual < best_residual) {
            best_residual = residual;
            best = candidate;
        }
    }
    return best;
}

SecondsPerTimeCount calibrate_time_model() {
    SecondsPerTimeCount model{};
    for (const Engine engine : {Engine::Dense,
                                Engine::Stabilizer,
                                Engine::Pauli,
                                Engine::MPS,
                                Engine::TensorNetwork,
                                Engine::DensityMatrix,
                                Engine::Trajectory}) {
        const SimulationEngine* const impl = cpu_engine(engine);
        if (impl == nullptr) {
            continue;
        }
        std::vector<Counts> rows;
        std::vector<double> seconds;
        try {
            for (const Case& c : cases_for(engine)) {
                const std::optional<std::pair<Counts, double>> m = measure(*impl, c);
                if (m.has_value() && m->second > 0.0) {
                    rows.push_back(m->first);
                    seconds.push_back(m->second);
                }
            }
        } catch (const std::exception&) {
            continue;  // this engine stays uncalibrated
        }
        if (!rows.empty()) {
            model[static_cast<std::size_t>(engine)] = fit_seconds_per_count(rows, seconds);
        }
    }
    return model;
}

}  // namespace menos::plan
