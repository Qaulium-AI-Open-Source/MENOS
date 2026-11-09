// Phase 24 step 1, item 1 (docs/adr/0028 sections 16.7 Q1/Q2 and 16.9): how
// close is every engine's predicted runtime to its measured runtime?
//
// For each workload the program asks the planner for the job's plan under the
// calibrated profile and a time budget too large to bind (so every candidate
// is converted to seconds), and takes the target engine's candidate's
// estimated_seconds as the prediction. It then runs that engine's plan:
//   - the planner's own plan when the target won;
//   - otherwise a Direct plan built by hand from the same reduced circuit with
//     plan::build_direct_region (the methodology of the other correlation
//     benchmarks). Factorized, cut and trajectory plans are only measured
//     when the planner chooses them, because they cannot be built by hand
//     faithfully.
// A candidate the planner pruned or did not convert has no estimated_seconds;
// its prediction is then computed from the engine's own estimate with the
// planner's CPU formula, max(work / rate, bytes / bandwidth), and the row says
// so (pred_source=formula).
//
// Actual is Result::execute_seconds() + Result::sample_seconds(); planning is
// not timed. One untimed warm-up, then kRepetitions timed runs; the row gives
// the median, minimum and maximum. Output is CSV on stdout, one row per
// workload, for bench/tools/accuracy_table.py.
//
// A report, not a unit test (CLAUDE.md: benchmarks are not tests).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/thread_count.h"
#include "menos/circuit.h"
#include "menos/noise_method.h"
#include "menos/observable.h"
#include "menos/plan.h"
#include "menos/result.h"
#include "menos/simulation_engine.h"
#include "menos_benchmark_environment.h"
#include "plan/available_engines.h"
#include "plan/hardware_profile.h"
#include "plan/job_spec.h"
#include "plan/planner.h"
#include "plan/region_builder.h"
#include "planner_corpus.h"
#include "runtime/runtime.h"

namespace {

using namespace menos;
namespace corpus = menos::planner_corpus;

constexpr int kRepetitions = 5;
// A budget no workload here comes near: it only makes the planner convert
// every candidate to seconds.
constexpr double kNonBindingBudgetSeconds = 1.0e6;
// Workloads predicted to take longer than this are skipped, so one run of the
// program stays bounded; the row still records the prediction.
constexpr double kMaxPredictedSeconds = 600.0;
// A workload whose untimed first run takes longer than this is not repeated:
// that first run is recorded as its only sample (the row says repetitions=1).
constexpr double kSingleRunSeconds = 10.0;

struct Workload {
    std::string engine_label;  // the engine (and backend) being measured
    std::string family;        // easy / representative / stress family name
    Strategy target;
    Backend backend = Backend::Cpu;
    plan::JobSpec spec;
};

std::string strategy_name(Strategy s) {
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

std::optional<Engine> direct_engine(Strategy s) {
    switch (s) {
        case Strategy::DirectDense:
            return Engine::Dense;
        case Strategy::DirectStabilizer:
            return Engine::Stabilizer;
        case Strategy::DirectPauli:
            return Engine::Pauli;
        case Strategy::DirectMPS:
            return Engine::MPS;
        case Strategy::DirectTensorNetwork:
            return Engine::TensorNetwork;
        case Strategy::DirectDensityMatrix:
            return Engine::DensityMatrix;
        default:
            return std::nullopt;
    }
}

const SimulationEngine* find_cpu_engine(Engine id) {
    for (const SimulationEngine* engine : plan::noisy_job_engines()) {
        if (engine->engine_id() == id && engine->backend() == Backend::Cpu &&
            engine->tier() == Tier::InMemory) {
            return engine;
        }
    }
    return nullptr;
}

// --- workloads (Q1) ----------------------------------------------------------

plan::JobSpec base(plan::JobSpec spec) {
    spec.memory_budget_bytes = menos_benchmark::safe_allocation_budget_bytes();
    spec.time_budget_seconds = kNonBindingBudgetSeconds;
    spec.max_threads = common::default_thread_count();
    return spec;
}

Circuit qft(std::uint32_t n) {
    Circuit circuit(n, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append(Opcode::X, {q});  // a non-trivial input state
    }
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
        for (std::uint32_t k = q + 1; k < n; ++k) {
            const double angle = 3.141592653589793 / static_cast<double>(1ULL << (k - q));
            circuit.append(Opcode::CP, {k, q}, {angle});
        }
    }
    return circuit;
}

Circuit ghz(std::uint32_t n) {
    Circuit circuit(n, n);
    circuit.append(Opcode::H, {0});
    for (std::uint32_t q = 0; q + 1 < n; ++q) {
        circuit.append(Opcode::CX, {q, q + 1});
    }
    corpus::measure_all(circuit);
    return circuit;
}

// H on every qubit, then CZ on `edges` random edges.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): n qubits, edges, a seed.
Circuit graph_state(std::uint32_t n, std::uint32_t edges, std::uint64_t seed) {
    corpus::SplitMix64 rng(seed);
    Circuit circuit(n, n);
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append(Opcode::H, {q});
    }
    for (std::uint32_t e = 0; e < edges; ++e) {
        const std::uint32_t a = rng.below(n);
        std::uint32_t b = rng.below(n - 1);
        if (b >= a) {
            ++b;
        }
        circuit.append(Opcode::CZ, {a, b});
    }
    corpus::measure_all(circuit);
    return circuit;
}

// First-order Trotter steps of the transverse-field Ising chain: RZZ on every
// edge, RX on every qubit. Small angles keep entanglement low (the MPS regime).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): n qubits, steps.
Circuit ising_trotter(std::uint32_t n, std::uint32_t steps, bool measured) {
    Circuit circuit(n, measured ? n : 0);
    for (std::uint32_t s = 0; s < steps; ++s) {
        for (std::uint32_t q = 0; q + 1 < n; ++q) {
            circuit.append(Opcode::RZZ, {q, q + 1}, {0.1});
        }
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RX, {q}, {0.05});
        }
    }
    if (measured) {
        corpus::measure_all(circuit);
    }
    return circuit;
}

// A Clifford circuit with `rotations` small-angle RZ gates sprinkled in: the
// Pauli propagation regime (Clifford gates plus single-qubit rotations).
// n, depth, rotations and a seed, named at every call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Circuit clifford_plus_rotations(std::uint32_t n,
                                std::uint32_t depth,
                                std::uint32_t rotations,
                                std::uint64_t seed) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    corpus::SplitMix64 rng(seed);
    Circuit circuit = corpus::random_clifford(n, depth, seed, false);
    for (std::uint32_t i = 0; i < rotations; ++i) {
        circuit.append(Opcode::RZ, {rng.below(n)}, {0.1 + 0.01 * static_cast<double>(i)});
        const std::uint32_t q = rng.below(n - 1);
        circuit.append(Opcode::CX, {q, q + 1});
    }
    return circuit;
}

// Two blocks of `width` qubits, each scrambled by random long-range pairs (hard
// for MPS), joined by one CX in the middle; 2 * width qubits are beyond the
// dense state, so a one-wire cut is the plan that fits.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): width, depth, a seed.
Circuit cut_scrambled(std::uint32_t width, std::uint32_t depth, std::uint64_t seed) {
    corpus::SplitMix64 rng(seed);
    const std::uint32_t n = 2 * width;
    Circuit circuit(n, 0);
    const auto layer = [&]() {
        for (std::uint32_t q = 0; q < n; ++q) {
            circuit.append(Opcode::RY, {q}, {rng.angle()});
        }
        for (std::uint32_t b = 0; b < 2; ++b) {
            for (std::uint32_t g = 0; g < width / 2; ++g) {
                const std::uint32_t a = rng.below(width);
                std::uint32_t c = rng.below(width - 1);
                if (c >= a) {
                    ++c;
                }
                circuit.append(Opcode::CX, {b * width + a, b * width + c});
            }
        }
    };
    for (std::uint32_t l = 0; l < depth / 2; ++l) {
        layer();
    }
    circuit.append(Opcode::CX, {width - 1, width});
    for (std::uint32_t l = depth / 2; l < depth; ++l) {
        layer();
    }
    return circuit;
}

// Two all-to-all CZ clusters of `per_cluster` qubits, after an RY layer, with a
// T on the second cluster's first qubit and qubit 0 then interacting with the
// whole second cluster: tests/api/cut_differential_test.cpp's fixture.
Circuit migrating_clusters(std::uint32_t per_cluster) {
    const std::uint32_t n = 2 * per_cluster;
    Circuit circuit(n, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        circuit.append(Opcode::RY, {q}, {0.21 + 0.07 * static_cast<double>(q)});
    }
    for (const std::uint32_t base : {0U, per_cluster}) {
        for (std::uint32_t a = base; a < base + per_cluster; ++a) {
            for (std::uint32_t b = a + 1; b < base + per_cluster; ++b) {
                circuit.append(Opcode::CZ, {a, b});
            }
        }
    }
    circuit.append(Opcode::T, {per_cluster});
    for (std::uint32_t b = per_cluster; b < n; ++b) {
        circuit.append(Opcode::CZ, {0, b});
    }
    return circuit;
}

// A Clifford block of `clifford_n` qubits beside a brickwork block of
// `dense_n` qubits, never interacting: a Factorized plan with two engines.
// Two widths, a depth and a seed, named at every call site.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Circuit mixed_blocks(std::uint32_t clifford_n,
                     std::uint32_t dense_n,
                     std::uint32_t depth,
                     std::uint64_t seed) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    corpus::SplitMix64 rng(seed);
    const std::uint32_t n = clifford_n + dense_n;
    Circuit circuit(n, n);
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (std::uint32_t q = 0; q < clifford_n; ++q) {
            circuit.append(rng.below(2) == 0 ? Opcode::H : Opcode::S, {q});
        }
        for (std::uint32_t q = layer % 2; q + 1 < clifford_n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
        for (std::uint32_t q = clifford_n; q < n; ++q) {
            circuit.append(Opcode::RY, {q}, {rng.angle()});
        }
        for (std::uint32_t q = clifford_n + layer % 2; q + 1 < n; q += 2) {
            circuit.append(Opcode::CX, {q, q + 1});
        }
    }
    corpus::measure_all(circuit);
    return circuit;
}

std::vector<Workload> workloads() {
    using corpus::counts_job;
    using corpus::expectation_job;
    using corpus::statevector_job;
    std::vector<Workload> w;
    const auto add = [&](std::string engine,
                         std::string family,
                         Strategy target,
                         plan::JobSpec spec,
                         Backend backend = Backend::Cpu,
                         std::optional<std::uint64_t> memory_budget = std::nullopt) {
        plan::JobSpec job = base(std::move(spec));
        if (memory_budget.has_value()) {
            job.memory_budget_bytes = memory_budget;
        }
        w.push_back(
            Workload{std::move(engine), std::move(family), target, backend, std::move(job)});
    };

    // Dense CPU: random/general, QFT, entangling.
    for (const std::uint32_t n : {16U, 20U, 22U, 24U}) {
        add("dense",
            "random_brickwork_d20/n" + std::to_string(n),
            Strategy::DirectDense,
            counts_job(corpus::brickwork(n, 20, 201, true)));
    }
    for (const std::uint32_t n : {16U, 20U, 22U}) {
        add("dense", "qft/n" + std::to_string(n), Strategy::DirectDense, statevector_job(qft(n)));
    }
    for (const std::uint32_t n : {20U, 24U}) {
        add("dense", "ghz/n" + std::to_string(n), Strategy::DirectDense, counts_job(ghz(n)));
    }
    // Dense CUDA: the same random family, backend forced (skipped without CUDA).
    for (const std::uint32_t n : {20U, 24U}) {
        plan::JobSpec spec = counts_job(corpus::brickwork(n, 20, 202, true));
        spec.backend = Backend::Cuda;
        add("dense_cuda",
            "random_brickwork_d20/n" + std::to_string(n),
            Strategy::DirectDense,
            std::move(spec),
            Backend::Cuda);
    }

    // Stabilizer: Clifford and graph states.
    for (const std::uint32_t n : {256U, 1000U}) {
        add("stabilizer",
            "random_clifford_d100/n" + std::to_string(n),
            Strategy::DirectStabilizer,
            counts_job(corpus::random_clifford(n, 100, 203, true)));
    }
    for (const std::uint32_t n : {500U, 2000U}) {
        add("stabilizer",
            "graph_state/n" + std::to_string(n),
            Strategy::DirectStabilizer,
            counts_job(graph_state(n, 4 * n, 204)));
    }

    // Pauli propagation: Pauli-heavy expectation workloads.
    for (const std::uint32_t rotations : {2U, 4U, 6U, 8U, 10U, 12U}) {
        add("pauli",
            "clifford_rz" + std::to_string(rotations) + "_paulis400/n40",
            Strategy::DirectPauli,
            expectation_job(clifford_plus_rotations(40, 20, rotations, 205),
                            corpus::random_paulis(40, 400, 206)));
    }

    // MPS: low and high entanglement.
    for (const std::uint32_t n : {16U, 32U, 64U}) {
        add("mps",
            "ising_trotter_s4/n" + std::to_string(n),
            Strategy::DirectMPS,
            counts_job(ising_trotter(n, 4, true)));
    }
    for (const std::uint32_t n : {12U, 16U, 20U}) {
        add("mps",
            "random_brickwork_d8/n" + std::to_string(n),
            Strategy::DirectMPS,
            counts_job(corpus::brickwork(n, 8, 207, true)));
    }

    // Tensor network: low and high treewidth.
    for (const std::uint32_t n : {30U, 40U}) {
        add("tensor_network",
            "brickwork_d4_zall/n" + std::to_string(n),
            Strategy::DirectTensorNetwork,
            expectation_job(corpus::brickwork(n, 4, 208, false), corpus::z_on_every_qubit(n)));
    }
    add("tensor_network",
        "random_pairs_d6_ising/n24",
        Strategy::DirectTensorNetwork,
        expectation_job(corpus::random_pairs(24, 6, 209, false), corpus::ising_chain(24)));

    // Cut: partitionable and highly connected.
    // Cut never beats MPS or the tensor network on time alone (nor in the
    // golden corpus); it is the plan that fits when memory is tight. These
    // workloads carry a 2 MiB memory budget, below the dense state and the MPS
    // peak, so the cut plan is the one the planner runs.
    for (const std::uint32_t width : {10U, 12U, 14U}) {
        plan::JobSpec spec =
            expectation_job(cut_scrambled(width, 8, 210), corpus::z_on_every_qubit(2 * width));
        add("cut",
            "cut_scrambled_d8_budget2MiB/w" + std::to_string(width),
            Strategy::CutAndRecombine,
            std::move(spec),
            Backend::Cpu,
            std::uint64_t{2} << 20);
    }
    // The cut_differential_test fixture: two all-to-all CZ clusters, one
    // qubit migrating between them, under a 1 MB memory budget. Here the cut
    // is the cheapest feasible plan (added after step 1, docs/adr/0028
    // section 16.12).
    for (const std::uint32_t per_cluster : {4U, 5U, 6U}) {
        add("cut",
            "clusters_migrating1_budget1MB/c" + std::to_string(per_cluster),
            Strategy::CutAndRecombine,
            expectation_job(migrating_clusters(per_cluster),
                            corpus::z_on_every_qubit(2 * per_cluster)),
            Backend::Cpu,
            std::uint64_t{1'000'000});
    }
    add("cut",
        "random_pairs_d10_zall/n20",
        Strategy::CutAndRecombine,
        expectation_job(corpus::random_pairs(20, 10, 211, false), corpus::z_on_every_qubit(20)));

    // Factorized: independent blocks, and mixed engines.
    add("factorized",
        "blocks_4x8_d30",
        Strategy::Factorized,
        counts_job(corpus::blocks_circuit(4, 8, 30, 212, true)));
    add("factorized",
        "blocks_2x18_d20",
        Strategy::Factorized,
        counts_job(corpus::blocks_circuit(2, 18, 20, 213, true)));
    add("factorized",
        "mixed_clifford200_dense18_d20",
        Strategy::Factorized,
        counts_job(mixed_blocks(200, 18, 20, 214)));

    // Noisy: density matrix and trajectories.
    for (const std::uint32_t n : {8U, 10U, 12U}) {
        add("density_matrix",
            "noisy_brickwork_d10/n" + std::to_string(n),
            Strategy::DirectDensityMatrix,
            counts_job(corpus::noisy_brickwork(n, 10, 215, 0.01)));
    }
    for (const std::uint32_t n : {12U, 14U}) {
        plan::JobSpec spec = counts_job(corpus::noisy_brickwork(n, 6, 216, 0.01));
        spec.noise_method = NoiseMethod::Auto;
        add("trajectory",
            "noisy_brickwork_d6/n" + std::to_string(n),
            Strategy::DirectTrajectory,
            std::move(spec));
    }
    return w;
}

// --- measurement --------------------------------------------------------------

struct Prediction {
    double seconds = -1.0;
    std::string source = {};  // "planner" or "formula"
    std::string note = {};
};

double formula_seconds(const CostEstimate& estimate, const plan::HardwareProfile& profile) {
    return std::max(
        static_cast<double>(estimate.estimated_work_units) / profile.work_units_per_second,
        static_cast<double>(estimate.estimated_peak_bytes) /
            profile.memory_bandwidth_bytes_per_second);
}

CostEstimate estimate_direct(const SimulationEngine& engine,
                             const plan::JobSpec& spec,
                             const Circuit& circuit) {
    const Observable* observable = spec.expectation.has_value() ? &*spec.expectation : nullptr;
    return engine.estimate(EngineEstimateRequest{.num_qubits = circuit.num_qubits(),
                                                 .operations = &circuit.operations(),
                                                 .want_statevector = spec.want_statevector,
                                                 .want_counts = spec.shots > 0,
                                                 .want_expectation = observable != nullptr,
                                                 .observable = observable,
                                                 .accuracy_policy = spec.accuracy_policy,
                                                 .budget_bytes = spec.memory_budget_bytes.value_or(
                                                     std::numeric_limits<std::uint64_t>::max()),
                                                 .shots = spec.shots,
                                                 .bitstring_width = circuit.num_clbits(),
                                                 .is_sole_region = true});
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 == 1 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

void run_workload(const Workload& w, const plan::HardwareProfile& profile) {
    const auto row = [&](const std::string& status,
                         const Prediction& p,
                         const std::string& won,
                         double med,
                         double lo,
                         double hi) {
        std::cout << w.engine_label << ',' << w.family << ',' << strategy_name(w.target) << ','
                  << (w.backend == Backend::Cuda ? "cuda" : "cpu") << ',' << status << ',' << won
                  << ',' << p.source << ',' << p.seconds << ',' << med << ',' << lo << ',' << hi
                  << ',' << (p.seconds > 0.0 && med > 0.0 ? med / p.seconds : -1.0) << ',' << '"'
                  << p.note << '"' << '\n';
    };

    Plan planned;
    try {
        planned = plan::plan_job(w.spec, profile);
    } catch (const std::exception& e) {
        row("plan_refused", Prediction{.note = e.what()}, "", -1, -1, -1);
        return;
    }

    const CandidateSummary* candidate = nullptr;
    for (const CandidateSummary& c : planned.candidates) {
        if (c.strategy == w.target && (c.is_winner || !candidate)) {
            candidate = &c;
        }
    }
    const bool won = planned.strategy == w.target &&
                     (planned.regions.empty() || planned.regions.front().backend == w.backend);

    Prediction prediction;
    Plan to_run;
    if (won) {
        prediction = Prediction{.seconds = candidate != nullptr ? candidate->estimated_seconds : -1,
                                .source = "planner"};
        to_run = planned;
    } else {
        const std::optional<Engine> engine_id = direct_engine(w.target);
        const SimulationEngine* engine =
            engine_id.has_value() ? find_cpu_engine(*engine_id) : nullptr;
        if (engine == nullptr || w.backend != Backend::Cpu) {
            row("not_chosen",
                Prediction{.seconds = candidate != nullptr ? candidate->estimated_seconds : -1,
                           .source = "planner",
                           .note = "winner " + strategy_name(planned.strategy) +
                                   (candidate != nullptr && !candidate->feasible
                                        ? "; target: " + candidate->infeasible_reason
                                        : "")},
                "no",
                -1,
                -1,
                -1);
            return;
        }
        const Circuit& circuit =
            planned.reduced_circuit.has_value() ? *planned.reduced_circuit : w.spec.circuit;
        const CostEstimate estimate = estimate_direct(*engine, w.spec, circuit);
        if (!estimate.feasible) {
            row("infeasible", Prediction{.note = estimate.infeasible_reason}, "no", -1, -1, -1);
            return;
        }
        if (candidate != nullptr && candidate->feasible && candidate->estimated_seconds > 0.0) {
            prediction = Prediction{.seconds = candidate->estimated_seconds, .source = "planner"};
        } else {
            prediction = Prediction{.seconds = formula_seconds(estimate, profile),
                                    .source = "formula",
                                    .note = candidate == nullptr
                                                ? "candidate not enumerated"
                                                : "candidate: " + candidate->infeasible_reason};
        }
        to_run = planned;
        to_run.strategy = w.target;
        to_run.regions.clear();
        const Observable* observable =
            w.spec.expectation.has_value() ? &*w.spec.expectation : nullptr;
        to_run.regions.push_back(plan::build_direct_region(
            circuit, engine->engine_id(), Backend::Cpu, estimate, observable));
        to_run.estimated_state_bytes = estimate.estimated_peak_bytes;
        to_run.estimated_work_units = estimate.estimated_work_units;
        to_run.cut_summary.reset();
        to_run.trajectory_summary.reset();
    }

    if (prediction.seconds > kMaxPredictedSeconds) {
        row("skipped_slow", prediction, won ? "yes" : "no", -1, -1, -1);
        return;
    }

    try {
        const runtime::Runtime rt;
        plan::JobSpec run_spec = w.spec;
        run_spec.time_budget_seconds.reset();               // no monitor: time the engine alone
        const Result first = rt.execute(to_run, run_spec);  // untimed warm-up
        const double first_seconds = first.execute_seconds() + first.sample_seconds();
        std::vector<double> times;
        if (first_seconds > kSingleRunSeconds) {
            times.push_back(first_seconds);
            prediction.note =
                "repetitions=1 (first run over " + std::to_string(kSingleRunSeconds) + " s)";
        } else {
            for (int i = 0; i < kRepetitions; ++i) {
                const Result result = rt.execute(to_run, run_spec);
                times.push_back(result.execute_seconds() + result.sample_seconds());
            }
        }
        row("measured",
            prediction,
            won ? "yes" : "no",
            median(times),
            *std::min_element(times.begin(), times.end()),
            *std::max_element(times.begin(), times.end()));
    } catch (const std::exception& e) {
        prediction.note = e.what();
        row("run_failed", prediction, won ? "yes" : "no", -1, -1, -1);
    }
}

}  // namespace

// Usage: menos_cost_model_accuracy_benchmark [engine_label]
// With an argument, only that engine's workloads run (for example "cut").
int main(int argc, char** argv) {
    const std::string only = argc > 1 ? std::string(argv[1]) : std::string();
    std::cerr << menos_benchmark::describe_environment() << "\n";
    const plan::HardwareProfile& profile = plan::calibrated_hardware_profile();
    std::cerr << "calibrated: work_units_per_second=" << profile.work_units_per_second
              << " memory_bandwidth_bytes_per_second=" << profile.memory_bandwidth_bytes_per_second
              << " cuda_device_work_units_per_second=" << profile.cuda_device_work_units_per_second
              << " threads=" << common::default_thread_count() << "\n";
    std::cout
        << "engine,workload,target,backend,status,won,pred_source,predicted_s,actual_median_s,"
           "actual_min_s,actual_max_s,ratio,note\n";
    for (const Workload& w : workloads()) {
        if (!only.empty() && w.engine_label != only) {
            continue;
        }
        run_workload(w, profile);
        std::cout.flush();
    }
    return 0;
}
