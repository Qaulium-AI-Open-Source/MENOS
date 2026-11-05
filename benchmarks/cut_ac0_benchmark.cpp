// Phase 9 AC-0: the pre-registered precondition measurement (docs/adr/0013
// section 10.1). This program needs NO cutting engine -- that is the point.
// A cut plan's cost is assembled from the EXISTING engines' own estimate()
// calls applied to the fragment circuits, and its runtime is measured by
// actually executing those fragments on the existing engines. So the question
// "is CutAndRecombine worth building" is answered before it is built.
//
// The circuit family, the matrix, and the five-part pass rule were all fixed
// in docs/adr/0013 before this ran. Nothing here may be tuned to produce a
// pass; a failure is a valid and useful outcome.

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/observable_restrict.h"
#include "common/rng.h"
#include "cost/cost_model.h"
#include "engines/dense/dense_engine.h"
#include "ir/wire_cut_basis.h"
#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/simulation_engine.h"
#include "menos_benchmark_environment.h"
#include "plan/available_engines.h"

namespace {

using namespace std::chrono;
using namespace menos;

constexpr std::uint64_t kRealisticBudget = 7'752'519'680ULL;  // docs/adr/0010's 50%-of-RAM figure
constexpr std::uint64_t kControlBudget = 64ULL * 1024 * 1024 * 1024;
constexpr std::uint64_t kSeed = 20260929;

// ---------------------------------------------------------------------------
// The (K,d)-clustered family, exactly as pre-registered.
//
// Two clusters of d = n/2 qubits with dense non-Clifford interaction inside
// each. K qubits MIGRATE: they interact inside cluster A for the first half,
// then inside cluster B for the second half. Those migration points are the
// wire cuts, so a K-cut plan exists by construction.
//
// This is the class Peng, Harrow, Ozols & Wu (arXiv:1904.00102) define their
// result for. It was chosen because it is the class the method targets, and
// fixed before measuring.
// ---------------------------------------------------------------------------

void dense_layer(Circuit& circuit,
                 const std::vector<QubitId>& qubits,
                 common::Rng& rng,
                 std::uint32_t depth) {
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        for (const QubitId q : qubits) {
            circuit.append(Opcode::RY, {q}, {rng.next_uniform() * 6.283185307179586});
            circuit.append(Opcode::RZ, {q}, {rng.next_uniform() * 6.283185307179586});
        }
        // All-to-all CZ within the set: this is what makes each cluster
        // individually hard (no thin interior cut for MPS, high line-graph
        // treewidth for the tensor network).
        for (std::size_t a = 0; a < qubits.size(); ++a) {
            for (std::size_t b = a + 1; b < qubits.size(); ++b) {
                circuit.append(Opcode::CZ, {qubits[a], qubits[b]});
            }
        }
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Circuit make_kd_clustered_circuit(std::uint32_t n, std::uint32_t k, std::uint32_t depth) {
    const std::uint32_t d = n / 2;
    common::Rng rng(kSeed);
    Circuit circuit(n, 0);

    std::vector<QubitId> cluster_a;
    std::vector<QubitId> cluster_b;
    std::vector<QubitId> migrating;
    for (std::uint32_t q = 0; q < d; ++q) {
        cluster_a.push_back(static_cast<QubitId>(q));
        if (q < k) {
            migrating.push_back(static_cast<QubitId>(q));
        }
    }
    for (std::uint32_t q = d; q < n; ++q) {
        cluster_b.push_back(static_cast<QubitId>(q));
    }

    // Phase 1: each cluster evolves independently.
    dense_layer(circuit, cluster_a, rng, depth);
    circuit.append(Opcode::T, {cluster_a.front()});
    dense_layer(circuit, cluster_b, rng, depth);
    circuit.append(Opcode::T, {cluster_b.front()});

    // Phase 2: the migrating qubits now belong to cluster B.
    std::vector<QubitId> b_plus = cluster_b;
    for (const QubitId q : migrating) {
        b_plus.push_back(q);
    }
    dense_layer(circuit, b_plus, rng, depth);
    return circuit;
}

// The two fragments the K cuts produce. Fragment A is cluster A with its
// phase-1 operations; fragment B is cluster B plus the migrated qubits, with
// its phase-1 and phase-2 operations.
struct Fragments {
    Circuit a;
    Circuit b;
    std::uint32_t k;
};

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Fragments make_fragments(std::uint32_t n, std::uint32_t k, std::uint32_t depth) {
    const std::uint32_t d = n / 2;
    common::Rng rng(kSeed);  // same seed, so fragments carry the same angles

    std::vector<QubitId> cluster_a;
    std::vector<QubitId> cluster_b;
    for (std::uint32_t q = 0; q < d; ++q) {
        cluster_a.push_back(static_cast<QubitId>(q));
    }
    for (std::uint32_t q = d; q < n; ++q) {
        cluster_b.push_back(static_cast<QubitId>(q));
    }

    // Fragment A: local qubits 0..d-1 map to cluster A.
    Circuit frag_a(d, 0);
    std::vector<QubitId> local_a;
    for (std::uint32_t q = 0; q < d; ++q) {
        local_a.push_back(static_cast<QubitId>(q));
    }
    dense_layer(frag_a, local_a, rng, depth);
    frag_a.append(Opcode::T, {0});

    // Fragment B: local qubits 0..d-1 map to cluster B, then k migrated
    // qubits at local d..d+k-1, each prepared by a Clifford at the cut.
    const std::uint32_t b_qubits = d + k;
    Circuit frag_b(b_qubits, 0);
    std::vector<QubitId> local_b;
    for (std::uint32_t q = 0; q < d; ++q) {
        local_b.push_back(static_cast<QubitId>(q));
    }
    // Preparation of the migrated wires: the worst case over the six states
    // is three Clifford gates (MinusI), used here so the cost is not
    // understated.
    for (std::uint32_t j = 0; j < k; ++j) {
        const auto wire = static_cast<QubitId>(d + j);
        for (const Opcode op : ir::preparation_gates(ir::CutPrepState::MinusI)) {
            frag_b.append(op, {wire});
        }
    }
    dense_layer(frag_b, local_b, rng, depth);
    frag_b.append(Opcode::T, {0});
    std::vector<QubitId> b_plus = local_b;
    for (std::uint32_t j = 0; j < k; ++j) {
        b_plus.push_back(static_cast<QubitId>(d + j));
    }
    dense_layer(frag_b, b_plus, rng, depth);

    return Fragments{frag_a, frag_b, k};
}

Observable z_on_every_qubit(std::uint32_t n) {
    PauliString p;
    p.num_qubits = n;
    p.x_words.assign((n + 63) / 64, 0);
    p.z_words.assign((n + 63) / 64, 0);
    for (std::uint32_t q = 0; q < n; ++q) {
        p.z_words[q / 64] |= (std::uint64_t{1} << (q % 64));
    }
    return Observable{{{p, {1.0, 0.0}}}};
}

double pow_u(double base, std::uint32_t exponent) {
    return std::pow(base, static_cast<double>(exponent));
}

struct CandidateResult {
    std::string name;
    bool feasible;
    std::uint64_t peak_bytes;
    std::uint64_t work;
    std::string reason;
};

std::vector<CandidateResult> evaluate_existing(const Circuit& circuit,
                                               const Observable& obs,
                                               std::uint64_t budget) {
    std::vector<CandidateResult> out;
    for (const SimulationEngine* engine : plan::available_engines()) {
        const CostEstimate e = cost::estimate_direct(circuit,
                                                     *engine,
                                                     /*want_statevector=*/false,
                                                     /*want_counts=*/false,
                                                     /*want_expectation=*/true,
                                                     &obs,
                                                     budget,
                                                     {},
                                                     /*trajectories=*/0,
                                                     /*shots=*/0);
        out.push_back(CandidateResult{std::string(engine->name()),
                                      e.feasible,
                                      e.estimated_peak_bytes,
                                      e.estimated_work_units,
                                      e.infeasible_reason.substr(0, 60)});
    }
    return out;
}

// The cut plan's own cost, assembled from docs/adr/0013 Eqs. (4.1)-(4.3)
// using the existing engines' estimates on the fragment circuits.
struct CutCost {
    bool feasible;
    std::uint64_t peak_bytes;  // (4.3) max over fragments, NOT the sum
    double work;               // (4.1) + (4.2)
    double distinct_evals;
    double branches;
    std::string reason;
};

CutCost evaluate_cut(const Fragments& frags, std::uint64_t budget) {
    const Observable obs_a = z_on_every_qubit(frags.a.num_qubits());
    const Observable obs_b = z_on_every_qubit(frags.b.num_qubits());

    CostEstimate best_a{false, 0, 0, "no engine"};
    CostEstimate best_b{false, 0, 0, "no engine"};
    for (const SimulationEngine* engine : plan::available_engines()) {
        const CostEstimate ea = cost::estimate_direct(frags.a,
                                                      *engine,
                                                      false,
                                                      false,
                                                      true,
                                                      &obs_a,
                                                      budget,
                                                      {},
                                                      /*trajectories=*/0,
                                                      /*shots=*/0);
        if (ea.feasible &&
            (!best_a.feasible || ea.estimated_work_units < best_a.estimated_work_units)) {
            best_a = ea;
        }
        const CostEstimate eb = cost::estimate_direct(frags.b,
                                                      *engine,
                                                      false,
                                                      false,
                                                      true,
                                                      &obs_b,
                                                      budget,
                                                      {},
                                                      /*trajectories=*/0,
                                                      /*shots=*/0);
        if (eb.feasible &&
            (!best_b.feasible || eb.estimated_work_units < best_b.estimated_work_units)) {
            best_b = eb;
        }
    }
    if (!best_a.feasible || !best_b.feasible) {
        return CutCost{false, 0, 0, 0, 0, "a fragment is itself infeasible"};
    }

    const double evals_a = pow_u(4.0, frags.k);  // 4 distinct observables per cut
    const double evals_b = pow_u(6.0, frags.k);  // 6 distinct prepared states per cut
    const double branches = pow_u(8.0, frags.k);
    const double work = evals_a * static_cast<double>(best_a.estimated_work_units) +
                        evals_b * static_cast<double>(best_b.estimated_work_units) + branches;
    const std::uint64_t peak = std::max(best_a.estimated_peak_bytes, best_b.estimated_peak_bytes);
    return CutCost{peak <= budget,
                   peak,
                   work,
                   evals_a + evals_b,
                   branches,
                   peak <= budget ? "" : "fragment peak above budget"};
}

}  // namespace

int main() {
    std::cout << menos_benchmark::describe_environment() << "\n";
    std::cout << "Phase 9 AC-0, pre-registered per docs/adr/0013 section 10.1\n";
    std::cout << "family: (K,d)-clustered, seed=" << kSeed << ", observable = Z on every qubit\n\n";

    for (const std::uint64_t budget : {kRealisticBudget, kControlBudget}) {
        std::cout << "================ budget = " << budget << " bytes ================\n";
        for (const std::uint32_t n : {20u, 24u, 28u, 32u}) {
            for (const std::uint32_t k : {1u, 2u, 3u, 4u, 8u}) {
                if (k > n / 2) {
                    continue;
                }
                constexpr std::uint32_t kDepth = 2;
                const Circuit full = make_kd_clustered_circuit(n, k, kDepth);
                const Observable obs = z_on_every_qubit(n);
                const std::vector<CandidateResult> existing = evaluate_existing(full, obs, budget);
                const Fragments frags = make_fragments(n, k, kDepth);
                const CutCost cut = evaluate_cut(frags, budget);

                std::cout << "n=" << n << " K=" << k << " d=" << (n / 2)
                          << " ops=" << full.operations().size() << "\n";
                for (const CandidateResult& c : existing) {
                    std::cout << "    " << std::left << std::setw(16) << c.name;
                    if (c.feasible) {
                        std::cout << "FEASIBLE peak=" << c.peak_bytes << " work=" << c.work << "\n";
                    } else {
                        std::cout << "infeasible (" << c.reason << ")\n";
                    }
                }
                std::cout << "    " << std::left << std::setw(16) << "CUT";
                if (cut.feasible) {
                    std::cout << "FEASIBLE peak=" << cut.peak_bytes << " work=" << cut.work
                              << " distinct_evals=" << cut.distinct_evals
                              << " branches=" << cut.branches << "\n";
                } else {
                    std::cout << "infeasible (" << cut.reason << ")\n";
                }

                // The AC-0 rule, evaluated mechanically rather than by eye.
                double cheapest_feasible = 0;
                bool any_feasible = false;
                for (const CandidateResult& c : existing) {
                    if (c.feasible &&
                        (!any_feasible || static_cast<double>(c.work) < cheapest_feasible)) {
                        cheapest_feasible = static_cast<double>(c.work);
                        any_feasible = true;
                    }
                }
                const bool cond1 = cut.feasible;
                const bool cond2 = cut.feasible && (!any_feasible || cut.work < cheapest_feasible);
                const bool cond3 =
                    cut.feasible && (!any_feasible || cut.work * 2.0 <= cheapest_feasible);
                std::cout << "    -> cond1(feasible)=" << cond1 << " cond2(cheapest)=" << cond2
                          << " cond3(2x margin)=" << cond3;
                if (any_feasible) {
                    std::cout << "  [cheapest existing work=" << cheapest_feasible << "]";
                } else {
                    std::cout << "  [no existing candidate feasible]";
                }
                std::cout << "\n\n";
            }
        }
    }
    // ---- AC-0 condition 4: MEASURED runtime, where direct is runnable ----
    // Executes the fragments on the existing dense engine and sums, against a
    // direct dense execution of the whole circuit. No cutting engine involved.
    std::cout << "\n======== condition 4: measured runtime (dense engine) ========\n";
    const engines::dense::DenseEngine dense_engine;
    for (const std::uint32_t n : {18u, 20u, 22u}) {
        constexpr std::uint32_t kDepth = 2;
        constexpr std::uint32_t kK = 1;
        const Circuit full = make_kd_clustered_circuit(n, kK, kDepth);
        const Observable obs = z_on_every_qubit(n);

        const auto run_one = [&](const Circuit& c, const Observable& o) {
            const EngineExecutionRequest req{
                .num_qubits = c.num_qubits(),
                .operations = c.operations(),
                .want_statevector = false,
                .want_counts = false,
                .want_expectation = true,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = 0,
                .global_phase = 0.0,
                .budget_bytes = kControlBudget,
                .is_sole_region = true,
                .observable = &o,
            };
            return dense_engine.execute(req);
        };

        const auto t0 = steady_clock::now();
        static_cast<void>(run_one(full, obs));
        const auto direct_us = duration_cast<microseconds>(steady_clock::now() - t0).count();

        const Fragments frags = make_fragments(n, kK, kDepth);
        const Observable obs_a = z_on_every_qubit(frags.a.num_qubits());
        const Observable obs_b = z_on_every_qubit(frags.b.num_qubits());
        const auto t1 = steady_clock::now();
        // 4 distinct measurement settings on fragment A, 6 distinct prepared
        // states on fragment B, per docs/adr/0013 Eq. (4.1).
        for (int i = 0; i < 4; ++i) {
            static_cast<void>(run_one(frags.a, obs_a));
        }
        for (int i = 0; i < 6; ++i) {
            static_cast<void>(run_one(frags.b, obs_b));
        }
        const auto cut_us = duration_cast<microseconds>(steady_clock::now() - t1).count();

        std::cout << "  n=" << n << " K=1  direct_us=" << direct_us << "  cut_us=" << cut_us
                  << "  speedup="
                  << (static_cast<double>(direct_us) /
                      static_cast<double>(cut_us == 0 ? 1 : cut_us))
                  << "x  cond4=" << (cut_us < direct_us) << "\n";
    }
    return 0;
}
