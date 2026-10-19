#include "memory_fixtures.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

#include "common/rng.h"
#include "engines/dense/dense_engine.h"
#include "engines/density_matrix/density_matrix_engine.h"
#include "engines/mps/mps_engine.h"
#include "engines/pauli/pauli_propagation_engine.h"
#include "engines/stabilizer/stabilizer_engine.h"
#include "engines/tensor_network/tensor_network_engine.h"
#include "engines/trajectory/trajectory_engine.h"
#include "heap_tracker.h"
#include "menos/menos.h"
#include "menos/mitigation.h"
#include "menos/simulation_engine.h"

#ifdef MENOS_MEMORY_WITH_CUDA
#include "engines/dense_cuda/cuda_dense_engine.h"
#endif

namespace menos::test::memory {
namespace {

using Complex = std::complex<double>;
using Kraus = std::vector<std::vector<Complex>>;

// ---- Circuits ---------------------------------------------------------------

// Qubit counts, depths and clbit counts are all std::uint32_t here; every call
// site below names what it passes.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)

// depth layers of RY on every qubit, a T on qubit 0 and a CX brick: non-Clifford
// and entangling, with fixed angles so every run is identical.
Circuit layered(std::uint32_t n, std::uint32_t depth, std::uint32_t num_clbits) {
    Circuit circuit(n, num_clbits);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(
                Opcode::RY,
                {q},
                {0.13 + (0.071 * static_cast<double>(q)) + (0.37 * static_cast<double>(layer))});
        }
        circuit.append(Opcode::T, {0});
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    return circuit;
}

Circuit clifford(std::uint32_t n, std::uint32_t depth, std::uint32_t num_clbits) {
    Circuit circuit(n, num_clbits);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append((q + layer) % 3 == 0 ? Opcode::S : Opcode::H, {q});
        }
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    return circuit;
}

// RZ is the Pauli-propagation engine's non-Clifford rotation (T is not
// supported there); `rotations` of them bound its term-count branching.
Circuit pauli_rotations(std::uint32_t n, std::uint32_t depth, std::uint32_t rotations) {
    Circuit circuit(n, 0);
    std::uint32_t placed = 0;
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::H, {q});
        }
        if (placed < rotations) {
            circuit.append(Opcode::RZ, {layer % n}, {0.29 + (0.1 * static_cast<double>(layer))});
            ++placed;
        }
        for (std::uint32_t q = layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    return circuit;
}

// Two clusters joined by one migrating qubit (tests/api/cut_differential_test.cpp).
Circuit clustered(std::uint32_t per_cluster) {
    const std::uint32_t n = 2 * per_cluster;
    Circuit circuit(n, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append(Opcode::RY, {q}, {0.21 + (0.07 * static_cast<double>(q))});
    }
    for (std::uint32_t a = 0; a < per_cluster; ++a) {
        for (std::uint32_t b = a + 1; b < per_cluster; ++b) {
            circuit.append(Opcode::CZ, {a, b});
        }
    }
    for (std::uint32_t a = per_cluster; a < n; ++a) {
        for (std::uint32_t b = a + 1; b < n; ++b) {
            circuit.append(Opcode::CZ, {a, b});
        }
    }
    circuit.append(Opcode::T, {per_cluster});
    for (std::uint32_t b = per_cluster; b < n; ++b) {
        circuit.append(Opcode::CZ, {0, b});
    }
    return circuit;
}

// Two blocks of `block` qubits that never interact: a Factorized plan. Every
// pair inside a block interacts, so neither the whole circuit nor a block is
// cheaper on MPS than on the dense engine.
Circuit two_blocks(std::uint32_t block, std::uint32_t depth, std::uint32_t num_clbits) {
    const std::uint32_t n = 2 * block;
    Circuit circuit(n, num_clbits);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(
                Opcode::RY,
                {q},
                {0.17 + (0.05 * static_cast<double>(q)) + (0.41 * static_cast<double>(layer))});
        }
        circuit.append(Opcode::T, {0});
        circuit.append(Opcode::T, {block});
        for (std::uint32_t base = 0; base < n; base += block) {
            for (std::uint32_t a = base; a < base + block; ++a) {
                for (std::uint32_t b = a + 1; b < base + block; ++b) {
                    circuit.append(Opcode::CZ, {a, b});
                }
            }
        }
    }
    return circuit;
}

Kraus depolarizing_1q(double p) {
    const double a = std::sqrt(1.0 - (0.75 * p));
    const double b = std::sqrt(p / 4.0);
    return {
        {{a, 0}, {0, 0}, {0, 0}, {a, 0}},
        {{0, 0}, {b, 0}, {b, 0}, {0, 0}},
        {{0, 0}, {0, -b}, {0, b}, {0, 0}},
        {{b, 0}, {0, 0}, {0, 0}, {-b, 0}},
    };
}

Circuit noisy(std::uint32_t n, std::uint32_t depth, std::uint32_t num_clbits) {
    Circuit circuit = layered(n, depth, num_clbits);
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append_channel({q}, depolarizing_1q(0.02));
    }
    return circuit;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void measure_all(Circuit& circuit, bool with_readout_error = false) {
    for (std::uint32_t q = 0; q < circuit.num_qubits(); ++q) {
        if (with_readout_error) {
            circuit.append_measure(q, q, ReadoutError{{0.97, 0.03, 0.05, 0.95}});
        } else {
            circuit.append_measure(q, q);
        }
    }
}

Circuit measured(Circuit circuit, bool with_readout_error = false) {
    measure_all(circuit, with_readout_error);
    return circuit;
}

// ---- Observables ------------------------------------------------------------

PauliString pauli(std::uint32_t n, const std::vector<std::pair<std::uint32_t, char>>& factors) {
    PauliString p;
    p.num_qubits = n;
    p.x_words.assign((n + 63) / 64, 0);
    p.z_words.assign((n + 63) / 64, 0);
    for (const auto& [q, op] : factors) {
        const std::uint64_t bit = std::uint64_t{1} << (q % 64);
        if (op == 'X' || op == 'Y') {
            p.x_words[q / 64] |= bit;
        }
        if (op == 'Z' || op == 'Y') {
            p.z_words[q / 64] |= bit;
        }
    }
    return p;
}

// Three terms over the first and last qubits.
Observable three_terms(std::uint32_t n) {
    Observable observable;
    observable.terms.emplace_back(pauli(n, {{0, 'Z'}}), Complex{0.5, 0.0});
    observable.terms.emplace_back(pauli(n, {{0, 'X'}, {1, 'X'}}), Complex{-0.25, 0.0});
    observable.terms.emplace_back(pauli(n, {{0, 'Z'}, {n - 1, 'Y'}}), Complex{0.125, 0.0});
    return observable;
}

Observable z_everywhere(std::uint32_t n) {
    std::vector<std::pair<std::uint32_t, char>> factors;
    factors.reserve(n);
    for (std::uint32_t q = 0; q < n; ++q) {
        factors.emplace_back(q, 'Z');
    }
    Observable observable;
    observable.terms.emplace_back(pauli(n, factors), Complex{1.0, 0.0});
    return observable;
}

Observable single_z(std::uint32_t n) {
    Observable observable;
    observable.terms.emplace_back(pauli(n, {{0, 'Z'}}), Complex{1.0, 0.0});
    return observable;
}

// ---- Fixture table ----------------------------------------------------------

Fixture engine_fixture(std::string name, std::string engine, Circuit circuit) {
    Fixture f{.name = std::move(name),
              .kind = FixtureKind::Engine,
              .engine = std::move(engine),
              .circuit = std::move(circuit)};
    return f;
}

Fixture run_fixture_of(std::string name, Circuit circuit, Strategy strategy) {
    Fixture f{.name = std::move(name),
              .kind = FixtureKind::Run,
              .engine = "run",
              .circuit = std::move(circuit)};
    f.expected_strategy = strategy;
    return f;
}

void add_dense_family(std::vector<Fixture>& out, const std::string& engine) {
    constexpr std::uint32_t kN = 18;
    {
        Fixture f = engine_fixture(engine + "_statevector", engine, layered(kN, 3, 0));
        f.want_statevector = true;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(engine + "_counts", engine, measured(layered(kN, 3, kN)));
        f.shots = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(engine + "_expectation", engine, layered(kN, 3, 0));
        f.observable = three_terms(kN);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(engine + "_all", engine, measured(layered(kN, 3, kN)));
        f.want_statevector = true;
        f.shots = 20000;
        f.observable = three_terms(kN);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(engine + "_region_counts", engine, measured(layered(16, 3, 16)));
        f.shots = 20000;
        f.sole_region = false;
        f.sampler_draws = 20000;
        out.push_back(std::move(f));
    }
    // The ADR 0025 Q9 gate size, probe only.
    {
        Fixture f = engine_fixture(engine + "_statevector_24", engine, layered(24, 2, 0));
        f.want_statevector = true;
        f.probe_only = true;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(engine + "_b", engine, measured(layered(1, 1, 1)));
        f.shots = 1;
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }
}

std::vector<Fixture> build_fixtures() {
    std::vector<Fixture> out;

    add_dense_family(out, "dense");
#ifdef MENOS_MEMORY_WITH_CUDA
    add_dense_family(out, "dense_cuda");
#endif

    // Density matrix: 4^n state, so n = 8.
    {
        Fixture f =
            engine_fixture("density_matrix_counts", "density_matrix", measured(noisy(8, 3, 8)));
        f.shots = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(
            "density_matrix_counts_readout", "density_matrix", measured(noisy(8, 3, 8), true));
        f.shots = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("density_matrix_expectation", "density_matrix", noisy(8, 3, 0));
        f.observable = three_terms(8);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(
            "density_matrix_region_counts", "density_matrix", measured(noisy(8, 3, 8)));
        f.shots = 20000;
        f.sole_region = false;
        f.sampler_draws = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("density_matrix_b", "density_matrix", measured(noisy(1, 1, 1)));
        f.shots = 1;
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }

    // Trajectories.
    {
        Fixture f =
            engine_fixture("trajectory_counts", "trajectory", measured(noisy(14, 3, 14), true));
        f.shots = 300;
        f.trajectories = 300;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("trajectory_expectation", "trajectory", noisy(14, 3, 0));
        f.observable = three_terms(14);
        f.trajectories = 300;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("trajectory_b", "trajectory", measured(noisy(1, 1, 1)));
        f.shots = 1;
        f.trajectories = 1;
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }

    // Stabilizer.
    {
        Fixture f =
            engine_fixture("stabilizer_counts", "stabilizer", measured(clifford(256, 6, 256)));
        f.shots = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("stabilizer_expectation", "stabilizer", clifford(256, 6, 0));
        f.observable = three_terms(256);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture(
            "stabilizer_region_counts", "stabilizer", measured(clifford(256, 6, 256)));
        f.shots = 20000;
        f.sole_region = false;
        f.sampler_draws = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("stabilizer_b", "stabilizer", measured(clifford(1, 1, 1)));
        f.shots = 1;
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }

    // Pauli propagation.
    {
        Fixture f = engine_fixture("pauli_expectation", "pauli", pauli_rotations(30, 8, 8));
        f.observable = three_terms(30);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("pauli_b", "pauli", pauli_rotations(1, 1, 1));
        f.observable = single_z(1);
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }

    // MPS: a nearest-neighbour brick, so bonds stay at most 2^depth.
    {
        Fixture f = engine_fixture("mps_counts", "mps", measured(layered(40, 5, 40)));
        f.shots = 2000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("mps_expectation", "mps", layered(40, 5, 0));
        f.observable = three_terms(40);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("mps_counts_controlled", "mps", measured(layered(40, 8, 40)));
        f.shots = 2000;
        f.accuracy_policy = AccuracyPolicy{
            .kind = AccuracyPolicyKind::Controlled, .tolerance = 1e-6, .max_bond_dimension = 16};
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("mps_region_counts", "mps", measured(layered(40, 5, 40)));
        f.shots = 2000;
        f.sole_region = false;
        f.sampler_draws = 2000;
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("mps_b", "mps", measured(layered(1, 1, 1)));
        f.shots = 1;
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }

    // Tensor network.
    {
        Fixture f =
            engine_fixture("tensor_network_expectation", "tensor_network", layered(14, 4, 0));
        f.observable = z_everywhere(14);
        out.push_back(std::move(f));
    }
    {
        Fixture f = engine_fixture("tensor_network_b", "tensor_network", layered(1, 1, 0));
        f.observable = single_z(1);
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }

    // Whole jobs through menos::run.
    {
        Fixture f =
            run_fixture_of("run_dense_statevector", layered(18, 3, 0), Strategy::DirectDense);
        f.want_statevector = true;
        out.push_back(std::move(f));
    }
    {
        Fixture f =
            run_fixture_of("run_factorized_statevector", two_blocks(9, 3, 0), Strategy::Factorized);
        f.want_statevector = true;
        out.push_back(std::move(f));
    }
    {
        Fixture f = run_fixture_of(
            "run_factorized_counts", measured(two_blocks(9, 3, 18)), Strategy::Factorized);
        f.shots = 20000;
        out.push_back(std::move(f));
    }
    {
        Fixture f =
            run_fixture_of("run_factorized_expectation", two_blocks(9, 3, 0), Strategy::Factorized);
        f.observable = three_terms(18);
        out.push_back(std::move(f));
    }
    {
        Fixture f = run_fixture_of("run_cut_expectation", clustered(4), Strategy::CutAndRecombine);
        f.observable = z_everywhere(8);
        out.push_back(std::move(f));
    }
    {
        // Whatever the planner picks for one qubit: B is the job's fixed cost.
        Fixture f = run_fixture_of("run_b", measured(layered(1, 1, 1)), Strategy::DirectDense);
        f.expected_strategy = std::nullopt;
        f.shots = 1;
        f.rss_baseline = true;
        out.push_back(std::move(f));
    }
    {
        Fixture f{.name = "run_mitigated",
                  .kind = FixtureKind::Mitigated,
                  .engine = "run",
                  .circuit = measured(layered(10, 3, 10), true)};
        f.shots = 20000;
        out.push_back(std::move(f));
    }
    return out;
}

// ---- Engines ----------------------------------------------------------------

std::unique_ptr<SimulationEngine> make_engine(const std::string& name) {
    if (name == "dense") {
        return std::make_unique<engines::dense::DenseEngine>();
    }
#ifdef MENOS_MEMORY_WITH_CUDA
    if (name == "dense_cuda") {
        return std::make_unique<engines::dense_cuda::CudaDenseEngine>();
    }
#endif
    if (name == "density_matrix") {
        return std::make_unique<engines::density_matrix::DensityMatrixEngine>();
    }
    if (name == "trajectory") {
        return std::make_unique<engines::trajectory::TrajectoryEngine>();
    }
    if (name == "stabilizer") {
        return std::make_unique<engines::stabilizer::StabilizerEngine>();
    }
    if (name == "pauli") {
        return std::make_unique<engines::pauli::PauliPropagationEngine>();
    }
    if (name == "mps") {
        return std::make_unique<engines::mps::MPSEngine>();
    }
    if (name == "tensor_network") {
        return std::make_unique<engines::tensor_network::TensorNetworkEngine>();
    }
    throw std::invalid_argument("unknown engine: " + name);
}

// The runtime's resolution of measurements: the last MeasureOp writing each
// clbit wins. Every fixture measures each qubit once, so this is just the list.
std::vector<std::pair<QubitId, ClbitId>> measure_targets(const Circuit& circuit) {
    std::map<ClbitId, QubitId> last;
    for (const Operation& op : circuit.operations()) {
        if (const auto* m = std::get_if<MeasureOp>(&op)) {
            last[m->clbit] = m->qubit;
        }
    }
    std::vector<std::pair<QubitId, ClbitId>> targets;
    targets.reserve(last.size());
    for (const auto& [clbit, qubit] : last) {
        targets.emplace_back(qubit, clbit);
    }
    return targets;
}

// ---- Digests ----------------------------------------------------------------

class Fnv1a {
public:
    void add(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash_ ^= bytes[i];
            hash_ *= 0x100000001b3ULL;
        }
    }
    void add(const std::string& s) {
        add(s.data(), s.size());
    }
    template <typename T>
    void add_value(const T& value) {
        add(&value, sizeof(value));
    }
    [[nodiscard]] std::string hex() const {
        std::array<char, 17> buffer{};
        std::snprintf(buffer.data(),
                      buffer.size(),
                      "%016llx",
                      static_cast<unsigned long long>(hash_));  // NOLINT(google-runtime-int)
        return {buffer.data()};
    }

private:
    std::uint64_t hash_ = 0xcbf29ce484222325ULL;
};

std::string digest_counts(const std::map<BitString, std::uint64_t>& counts) {
    Fnv1a h;
    for (const auto& [key, count] : counts) {
        h.add(key.to_hex_string());
        h.add_value(count);
    }
    return h.hex();
}

std::string digest_complex(const std::vector<Complex>& values) {
    Fnv1a h;
    for (const Complex& v : values) {
        const double re = v.real();
        const double im = v.imag();
        h.add_value(re);
        h.add_value(im);
    }
    return h.hex();
}

// ---- Execution --------------------------------------------------------------

Measurement run_engine_fixture(const Fixture& fixture) {
    const std::unique_ptr<SimulationEngine> engine = make_engine(fixture.engine);
    const Circuit& circuit = fixture.circuit;
    const bool want_counts = fixture.shots > 0;
    const bool want_expectation = fixture.observable.has_value();
    const Observable* observable = want_expectation ? &*fixture.observable : nullptr;

    const EngineEstimateRequest estimate_request{circuit.num_qubits(),
                                                 &circuit.operations(),
                                                 fixture.want_statevector,
                                                 want_counts,
                                                 want_expectation,
                                                 observable,
                                                 fixture.accuracy_policy,
                                                 std::numeric_limits<std::uint64_t>::max(),
                                                 fixture.trajectories};
    const CostEstimate estimate = engine->estimate(estimate_request);
    if (!estimate.feasible) {
        throw std::runtime_error(fixture.name +
                                 ": estimate infeasible: " + estimate.infeasible_reason);
    }

    const std::vector<std::pair<QubitId, ClbitId>> targets = measure_targets(circuit);
    EngineExecutionRequest request{
        .num_qubits = circuit.num_qubits(),
        .operations = circuit.operations(),
        .want_statevector = fixture.want_statevector,
        .want_counts = want_counts,
        .want_expectation = want_expectation,
        .shots = fixture.shots,
        .measure_targets = targets,
        .bitstring_width = circuit.num_clbits(),
        .seed = 2025,
        .global_phase = circuit.global_phase,
        // The runtime passes the region's own estimate as its budget.
        .budget_bytes = estimate.estimated_peak_bytes,
        .is_sole_region = fixture.sole_region,
        .observable = observable,
        .accuracy_policy = fixture.accuracy_policy,
        .monitor = nullptr,
        .trajectories = fixture.trajectories,
    };
    // The CUDA engine's host figure is 0 for an expectation-only job, which
    // allocates nothing on the host to budget.
    if (request.budget_bytes == 0) {
        request.budget_bytes = std::numeric_limits<std::uint64_t>::max();
    }

    Measurement m;
    m.estimated_peak_bytes = estimate.estimated_peak_bytes;
    m.estimated_device_bytes = estimate.device.device_bytes;
    m.strategy = fixture.engine;

    const std::uint64_t live_before = heap_tracker::live_bytes();
    heap_tracker::reset_peak();
    {
        const EngineExecutionResult result = engine->execute(request);
        std::map<BitString, std::uint64_t> drawn;
        if (result.counts_sampler) {
            common::Rng rng(99);
            for (std::uint64_t i = 0; i < fixture.sampler_draws; ++i) {
                drawn[result.counts_sampler->sample_one(rng)] += 1;
            }
        }
        if (heap_tracker::available()) {
            m.tracked_peak_bytes = heap_tracker::peak_bytes() - live_before;
        }
        m.reported_actual_bytes = result.actual_bytes;
        if (result.counts.has_value()) {
            m.counts_digest = digest_counts(*result.counts);
        } else if (result.counts_sampler) {
            m.counts_digest = digest_counts(drawn);
        }
        if (result.statevector.has_value()) {
            m.statevector_digest = digest_complex(*result.statevector);
        }
        if (result.expectation_terms.has_value()) {
            m.expectation_digest = digest_complex(*result.expectation_terms);
        }
    }
    return m;
}

RunOptions options_for(const Fixture& fixture) {
    RunOptions options{
        .shots = fixture.shots, .want_statevector = fixture.want_statevector, .seed = 2025};
    options.expectation = fixture.observable;
    options.accuracy_policy = fixture.accuracy_policy;
    return options;
}

std::uint64_t winner_peak_bytes(const Plan& plan) {
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.is_winner) {
            return candidate.estimated_peak_bytes;
        }
    }
    return plan.estimated_state_bytes;
}

std::uint64_t winner_device_bytes(const Plan& plan) {
    for (const CandidateSummary& candidate : plan.candidates) {
        if (candidate.is_winner) {
            return candidate.device.device_bytes;
        }
    }
    return 0;
}

const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::DirectDense:
            return "DirectDense";
        case Strategy::DirectStabilizer:
            return "DirectStabilizer";
        case Strategy::DirectPauli:
            return "DirectPauli";
        case Strategy::DirectMPS:
            return "DirectMPS";
        case Strategy::DirectTensorNetwork:
            return "DirectTensorNetwork";
        case Strategy::Factorized:
            return "Factorized";
        case Strategy::CutAndRecombine:
            return "CutAndRecombine";
        case Strategy::DirectDensityMatrix:
            return "DirectDensityMatrix";
        case Strategy::DirectTrajectory:
            return "DirectTrajectory";
    }
    return "?";
}

void record_result(const Result& result, Measurement& m) {
    m.reported_actual_bytes = result.actual_state_bytes();
    m.strategy = strategy_name(result.plan().strategy);
    if (!result.counts().empty()) {
        m.counts_digest = digest_counts(result.counts());
    }
    if (result.statevector().has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        m.statevector_digest = digest_complex(*result.statevector());
    }
    if (result.expectation_value().has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        m.expectation_digest = digest_complex({*result.expectation_value()});
    }
}

Measurement run_job_fixture(const Fixture& fixture) {
    const RunOptions options = options_for(fixture);
    prepare(fixture);

    Measurement m;
    const std::uint64_t live_before = heap_tracker::live_bytes();
    heap_tracker::reset_peak();
    if (fixture.kind == FixtureKind::Run) {
        const Result result = run(fixture.circuit, options);
        if (heap_tracker::available()) {
            m.tracked_peak_bytes = heap_tracker::peak_bytes() - live_before;
        }
        m.estimated_peak_bytes = winner_peak_bytes(result.plan());
        m.estimated_device_bytes = winner_device_bytes(result.plan());
        record_result(result, m);
        if (fixture.expected_strategy.has_value() &&
            result.plan().strategy != *fixture.expected_strategy) {
            throw std::runtime_error(fixture.name + ": planned " + m.strategy + ", expected " +
                                     strategy_name(*fixture.expected_strategy));
        }
    } else {
        const MitigatedResult mitigated = run_readout_mitigated(
            fixture.circuit, std::nullopt, options, ReadoutMitigationOptions{20000});
        if (heap_tracker::available()) {
            m.tracked_peak_bytes = heap_tracker::peak_bytes() - live_before;
        }
        // The three runs are held together until mitigation finishes.
        m.estimated_peak_bytes = winner_peak_bytes(mitigated.raw().plan());
        for (const Result& calibration : mitigated.calibration_results()) {
            m.estimated_peak_bytes += winner_peak_bytes(calibration.plan());
        }
        record_result(mitigated.raw(), m);
        Fnv1a h;
        for (const double q : mitigated.quasi_probabilities()) {
            h.add_value(q);
        }
        m.expectation_digest = h.hex();
    }
    return m;
}

}  // namespace

const std::vector<Fixture>& fixtures() {
    static const std::vector<Fixture> kFixtures = build_fixtures();
    return kFixtures;
}

const Fixture* find_fixture(std::string_view name) {
    for (const Fixture& f : fixtures()) {
        if (f.name == name) {
            return &f;
        }
    }
    return nullptr;
}

const Fixture& rss_baseline_for(const Fixture& fixture) {
    const std::string name = fixture.engine == "run" ? "run_b" : fixture.engine + "_b";
    const Fixture* baseline = find_fixture(name);
    if (baseline == nullptr) {
        throw std::invalid_argument("no RSS baseline fixture " + name);
    }
    return *baseline;
}

bool can_run(const Fixture& fixture) {
#ifdef MENOS_MEMORY_WITH_CUDA
    if (fixture.engine == "dense_cuda") {
        return engines::dense_cuda::cuda_device_available();
    }
#else
    static_cast<void>(fixture);
#endif
    return true;
}

void prepare(const Fixture& fixture) {
    if (fixture.kind != FixtureKind::Engine) {
        // Planning calibrates the machine once per process.
        static_cast<void>(preview(fixture.circuit, options_for(fixture)));
    }
}

Measurement run_fixture(const Fixture& fixture) {
    if (fixture.kind == FixtureKind::Engine) {
        return run_engine_fixture(fixture);
    }
    return run_job_fixture(fixture);
}

}  // namespace menos::test::memory
