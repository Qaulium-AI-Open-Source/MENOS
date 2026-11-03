#include "runtime/runtime.h"

#include <array>
#include <chrono>
#include <complex>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "analysis/cut.h"
#include "common/error.h"
#include "common/rng.h"
#include "common/splitmix64.h"
#include "cost/cost_model.h"
#include "ir/final_writes.h"
#include "plan/available_engines.h"
#include "plan/execution_engines.h"
#include "runtime/check_plan_invariants.h"
#include "runtime/combine_counts.h"
#include "runtime/combine_expectation.h"
#include "runtime/recombine_cut.h"
#include "runtime/recombine_statevector.h"

namespace menos::runtime {
namespace {

// Resolves a region's (engine, backend) pair. Both halves matter from Phase 10
// on: Engine::Dense alone no longer identifies an implementation, since the CPU
// and CUDA dense engines share it and differ only in backend.
const SimulationEngine& engine_for(Engine engine, Backend backend) {
    const SimulationEngine* found = plan::find_execution_engine(engine, backend);
    if (found == nullptr) {
        throw common::InternalError(
            "engine_for: this build has no engine for the plan's (engine, backend) pair -- a "
            "plan selecting a backend the runtime cannot execute is an internal invariant "
            "violation");
    }
    return *found;
}

Result execute_direct(const Plan& plan,
                      const menos::plan::JobSpec& job_spec,
                      const ExecutionMonitor* monitor) {
    common::ensure(plan.reduced_circuit.has_value(),
                   "execute_direct: plan.reduced_circuit must be set by plan_job -- internal "
                   "invariant violated");
    // common::ensure above already guarantees this; clang-tidy can't see
    // through it.
    const Circuit& reduced_circuit =
        *plan.reduced_circuit;  // NOLINT(bugprone-unchecked-optional-access)
    const Region& region = plan.regions[0];
    const SimulationEngine& engine = engine_for(region.engine, region.backend);
    const bool want_expectation = job_spec.expectation.has_value();

    // final_writes must be computed from the light-cone-REDUCED circuit,
    // not the job's original one: region.global_to_local is indexed by
    // the reduced circuit's own qubit numbering (docs/adr/0007), which
    // differs from the original whenever light-cone reduction actually
    // dropped or renumbered qubits (only a want_statevector request
    // guarantees no reduction -- src/plan/planner.cpp's
    // compute_light_cone_support). A measured qubit that reduction
    // dropped cannot appear in the reduced circuit's own MeasureOps at
    // all, so every write found here is already in-region by
    // construction; the local lookup below is a direct index, not a
    // filter.
    const std::map<ClbitId, QubitId> final_writes = ir::compute_final_writes(reduced_circuit);
    std::vector<std::pair<QubitId, ClbitId>> measure_targets;
    measure_targets.reserve(final_writes.size());
    for (const auto& [clbit, qubit] : final_writes) {
        const QubitId local = region.global_to_local[qubit];
        if (local != kNoLocalQubit) {
            measure_targets.emplace_back(local, clbit);
        }
    }

    const EngineExecutionRequest request{
        .num_qubits = static_cast<std::uint32_t>(region.local_to_global.size()),
        .operations = region.operations,
        .want_statevector = job_spec.want_statevector,
        .want_counts = job_spec.shots > 0,
        .want_expectation = want_expectation,
        .shots = job_spec.shots,
        .measure_targets = measure_targets,
        .bitstring_width = reduced_circuit.num_clbits(),
        .seed = job_spec.seed,
        // The ORIGINAL circuit's global_phase, never the reduced one's
        // (light-cone reduction always drops it to 0.0, docs/adr/0007 --
        // it cancels in every output this phase can request, but a
        // Direct* strategy still needs the true value here since this
        // region's engine is the whole job's sole region).
        .global_phase = job_spec.circuit.global_phase,
        .budget_bytes = region.estimated_state_bytes,
        .is_sole_region = true,
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        .observable = want_expectation ? &*region.observable : nullptr,
        .accuracy_policy = job_spec.accuracy_policy,
        .monitor = monitor,
        // The count the plan fixed (Phase 13); 0 for every non-trajectory plan.
        .trajectories =
            plan.trajectory_summary.has_value() ? plan.trajectory_summary->trajectories : 0,
        .max_threads = job_spec.max_threads,
        .fuse_diagonal_gates = job_spec.fuse_diagonal_gates,
    };

    const auto start_execute = std::chrono::steady_clock::now();
    EngineExecutionResult result = engine.execute(request);
    const double execute_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_execute).count();

    std::optional<std::complex<double>> expectation_value;
    if (want_expectation) {
        // A single sole region is a degenerate one-region "product":
        // combine_expectation's product-then-sum collapses to a plain
        // per-term coefficient-weighted sum here.
        const std::array<std::vector<std::complex<double>>, 1> per_region{
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            *result.expectation_terms};
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        expectation_value = combine_expectation(*region.observable, per_region);
    }

    // The actual, post-execution truth (CLAUDE.md's Controlled(tolerance)
    // domain invariant: "a run is exact if and only if no singular value
    // was discarded at any step"), not plan.exact's own pre-execution
    // estimate -- the two agree whenever plan.exact was already true
    // (every non-MPS engine, and MPS under Exact, never sets
    // accumulated_error_bound at all), and only differ when a
    // Controlled-policy MPS run happens not to have needed any
    // truncation after all.
    // A statistical engine (trajectories, Phase 13) is never exact, whatever it
    // sampled.
    const bool actually_exact = !result.accumulated_error_bound.has_value() && !result.statistical;
    Plan executed_plan = plan;
    executed_plan.regions[0].mps_accumulated_error_bound = result.accumulated_error_bound;
    if (region.engine == Engine::TensorNetwork) {
        executed_plan.regions[0].tensor_network_actual_peak_bytes = result.actual_bytes;
    }
    if (executed_plan.trajectory_summary.has_value()) {
        executed_plan.trajectory_summary->standard_error = result.expectation_standard_error;
    }

    return Result(result.counts.value_or(std::map<BitString, std::uint64_t>{}),
                  std::move(result.statevector),
                  expectation_value,
                  actually_exact,
                  std::move(executed_plan),
                  result.actual_bytes,
                  execute_seconds,
                  /*sample_seconds=*/0.0,
                  /*replans=*/{},
                  result.expectation_standard_error);
}

// {1, 0}: a static, never-allocated |0> amplitude pair for an idle
// region's single qubit (docs/BUILD_PLAN.md Phase 3, 3.2).
constexpr std::complex<double> kIdleAmplitudes[2] = {{1.0, 0.0}, {0.0, 0.0}};

// An idle region's own raw per-term expectation, with no engine
// invoked: an idle qubit stays exactly |0>, so <0...0|P|0...0> is 1 for
// a term whose restriction to this region is the identity (no X or Y
// component, i.e. every x-bit is 0) and 0 otherwise (docs/adr/0007,
// same "x!=0 is orthogonal to |0>" fact the Pauli propagation engine's
// own final evaluation step uses).
std::vector<std::complex<double>> idle_region_expectation_terms(
    const Observable& region_observable) {
    std::vector<std::complex<double>> terms;
    terms.reserve(region_observable.terms.size());
    for (const auto& term : region_observable.terms) {
        bool nonzero_x = false;
        for (std::uint64_t word : term.first.x_words) {
            if (word != 0) {
                nonzero_x = true;
                break;
            }
        }
        terms.emplace_back(nonzero_x ? 0.0 : 1.0, 0.0);
    }
    return terms;
}

Result execute_factorized(const Plan& plan,
                          const menos::plan::JobSpec& job_spec,
                          const ExecutionMonitor* monitor) {
    common::ensure(plan.reduced_circuit.has_value(),
                   "execute_factorized: plan.reduced_circuit must be set by plan_job -- internal "
                   "invariant violated");
    // common::ensure above already guarantees this; clang-tidy can't see
    // through it.
    const Circuit& reduced_circuit =
        *plan.reduced_circuit;  // NOLINT(bugprone-unchecked-optional-access)
    check_plan_invariants(reduced_circuit, plan);

    const bool want_expectation = job_spec.expectation.has_value();
    std::vector<std::vector<std::complex<double>>> per_region_expectation_terms;
    if (want_expectation) {
        per_region_expectation_terms.reserve(plan.regions.size());
    }

    const std::uint32_t bitstring_width = reduced_circuit.num_clbits();
    // Every Region::global_to_local below is indexed by the reduced
    // circuit's own qubit numbering (docs/adr/0007), so final_writes
    // must be resolved against it too, not job_spec.circuit.
    const std::map<ClbitId, QubitId> final_writes = ir::compute_final_writes(reduced_circuit);

    // Route each surviving (globally last-write-wins) (qubit, clbit) pair
    // to its owning region, by local qubit id.
    std::vector<std::vector<std::pair<QubitId, ClbitId>>> region_measure_targets(
        plan.regions.size());
    for (const auto& [clbit, qubit] : final_writes) {
        for (std::size_t r = 0; r < plan.regions.size(); ++r) {
            const QubitId local = plan.regions[r].global_to_local[qubit];
            if (local != kNoLocalQubit) {
                region_measure_targets[r].emplace_back(local, clbit);
                break;
            }
        }
    }

    std::optional<std::vector<std::complex<double>>> global;
    if (job_spec.want_statevector) {
        global =
            allocate_global_statevector(reduced_circuit.num_qubits(), plan.estimated_state_bytes);
    }

    std::vector<std::unique_ptr<RegionCountsSampler>> samplers;
    std::vector<common::Rng> rngs;
    common::SplitMix64 seed_mixer(job_spec.seed);

    const std::uint64_t global_bytes =
        global.has_value()
            ? static_cast<std::uint64_t>(global->size()) * sizeof(std::complex<double>)
            : 0;
    std::uint64_t actual_peak_bytes = global_bytes;

    // A mutable copy so each region's own post-execution
    // mps_accumulated_error_bound can be recorded (CLAUDE.md's
    // Controlled(tolerance) domain invariant: this is only known after
    // execution), mirroring execute_direct's identical pattern. Also
    // where the plan-time exactness estimate gets upgraded to the
    // actual, post-execution truth below.
    Plan executed_plan = plan;
    bool all_regions_exact = true;

    const auto start_execute = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < plan.regions.size(); ++r) {
        const std::uint64_t sub_seed = seed_mixer.next();
        const Region& region = plan.regions[r];
        const bool needs_counts = job_spec.shots > 0 && !region_measure_targets[r].empty();

        std::span<const std::complex<double>> psi_r;
        std::optional<EngineExecutionResult> engine_result;
        if (region.estimated_state_bytes == 0) {
            // Idle: statically |0>, no allocation, no engine invoked
            // (docs/BUILD_PLAN.md Phase 3, 3.2). No counts draw needed
            // even if measured -- the outcome is deterministic 0, and
            // combine_counts's BitString already starts at 0 for every
            // bit no region sets.
            psi_r = std::span<const std::complex<double>>(kIdleAmplitudes, 2);
            if (want_expectation) {
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                const Observable& region_observable = *region.observable;
                per_region_expectation_terms.push_back(
                    idle_region_expectation_terms(region_observable));
            }
        } else {
            const SimulationEngine& engine = engine_for(region.engine, region.backend);
            // global_phase is 0.0 here and applied once, as a
            // post-multiply over the whole global buffer below -- never
            // per region (docs/adr/0005's global-phase bug fix; unchanged
            // by Phase 4).
            const EngineExecutionRequest request{
                .num_qubits = static_cast<std::uint32_t>(region.local_to_global.size()),
                .operations = region.operations,
                .want_statevector = job_spec.want_statevector,
                .want_counts = needs_counts,
                .want_expectation = want_expectation,
                .shots = job_spec.shots,
                .measure_targets = region_measure_targets[r],
                .bitstring_width = bitstring_width,
                .seed = sub_seed,
                .global_phase = 0.0,
                .budget_bytes = region.estimated_state_bytes,
                .is_sole_region = false,
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                .observable = want_expectation ? &*region.observable : nullptr,
                .accuracy_policy = job_spec.accuracy_policy,
                .monitor = monitor,
                .max_threads = job_spec.max_threads,
                .fuse_diagonal_gates = job_spec.fuse_diagonal_gates,
            };
            engine_result = engine.execute(request);
            if (request.want_statevector) {
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                psi_r = std::span<const std::complex<double>>(*engine_result->statevector);
            }
            if (want_expectation) {
                std::vector<std::complex<double>> region_terms =
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    std::move(*engine_result->expectation_terms);
                per_region_expectation_terms.push_back(std::move(region_terms));
            }
            executed_plan.regions[r].mps_accumulated_error_bound =
                engine_result->accumulated_error_bound;
            if (region.engine == Engine::TensorNetwork) {
                executed_plan.regions[r].tensor_network_actual_peak_bytes =
                    engine_result->actual_bytes;
            }
            if (engine_result->accumulated_error_bound.has_value()) {
                all_regions_exact = false;
            }
        }
        actual_peak_bytes =
            std::max(actual_peak_bytes, global_bytes + region.estimated_state_bytes);

        if (global.has_value()) {
            multiply_region_into_global(*global, region, psi_r);
        }
        if (needs_counts && engine_result.has_value()) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            samplers.push_back(std::move(engine_result->counts_sampler));
            rngs.emplace_back(sub_seed);
        }
    }

    // Apply global_phase exactly once, as a scalar multiplying the whole
    // recombined state -- the ORIGINAL circuit's phase, never the
    // reduced one's (light-cone reduction always drops it to 0.0,
    // docs/adr/0007).
    if (global.has_value()) {
        const std::complex<double> phase =
            std::exp(std::complex<double>{0.0, job_spec.circuit.global_phase});
        for (std::complex<double>& amplitude : *global) {
            amplitude *= phase;
        }
    }

    const double execute_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_execute).count();

    std::map<BitString, std::uint64_t> counts;
    double sample_seconds = 0.0;
    if (job_spec.shots > 0) {
        std::vector<const RegionCountsSampler*> sampler_ptrs;
        sampler_ptrs.reserve(samplers.size());
        for (const std::unique_ptr<RegionCountsSampler>& sampler : samplers) {
            sampler_ptrs.push_back(sampler.get());
        }
        const auto start_sample = std::chrono::steady_clock::now();
        counts = combine_counts(sampler_ptrs, rngs, bitstring_width, job_spec.shots, monitor);
        sample_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_sample).count();
    }

    std::optional<std::vector<std::complex<double>>> statevector_out = std::move(global);

    std::optional<std::complex<double>> expectation_value;
    if (want_expectation) {
        common::ensure(!plan.regions.empty(),
                       "execute_factorized: a Factorized plan must have at least one region");
        // Every region's own restricted observable carries the SAME
        // original per-term coefficients (common::restrict_observable
        // never touches them), so any one of them is an equally valid
        // source for combine_expectation's coefficient argument.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        const Observable& coefficient_source = *plan.regions[0].observable;
        expectation_value = combine_expectation(coefficient_source, per_region_expectation_terms);
    }

    return Result(std::move(counts),
                  std::move(statevector_out),
                  expectation_value,
                  all_regions_exact,
                  std::move(executed_plan),
                  actual_peak_bytes,
                  execute_seconds,
                  sample_seconds);
}

}  // namespace

// Phase 9 (docs/adr/0013): run each fragment once per DISTINCT assignment on
// an existing engine, then reconstruct each observable term by the signed,
// weighted sum over the 8^K branches. There is no second simulator here: every
// fragment goes through the same SimulationEngine interface every other
// strategy uses.
Result execute_cut(const Plan& plan,
                   const menos::plan::JobSpec& job_spec,
                   const ExecutionMonitor* monitor) {
    common::ensure(plan.reduced_circuit.has_value(),
                   "execute_cut: plan.reduced_circuit must be set by plan_job");
    common::ensure(!plan.regions.empty() && plan.regions[0].observable.has_value(),
                   "execute_cut: a cut plan carries the job's own remapped observable on its "
                   "regions -- internal invariant violated");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const Circuit& reduced_circuit = *plan.reduced_circuit;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const Observable& observable = *plan.regions[0].observable;

    // Recomputed rather than carried on Plan, so no analysis type leaks into
    // the public API. find_cut_plan is deterministic, and the planner used the
    // same circuit and the same kMaxCuts, so this reproduces its decomposition.
    const std::optional<analysis::CutPlan> maybe_cut_plan =
        analysis::find_cut_plan(reduced_circuit, analysis::kMaxCuts);
    common::ensure(maybe_cut_plan.has_value(),
                   "execute_cut: the planner chose CutAndRecombine but no cut plan was found on "
                   "the same circuit -- internal invariant violated");
    // .value() rather than operator*: the ensure() above already guarantees
    // this, and the checked accessor keeps it honest if that ever changes.
    // clang-tidy cannot see through common::ensure, and flags .value() as well
    // as operator*, so the suppression is needed either way.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const analysis::CutPlan& cut_plan = maybe_cut_plan.value();

    const auto start_execute = std::chrono::steady_clock::now();
    const std::size_t term_count = observable.terms.size();
    const std::size_t fragment_count = cut_plan.fragments.size();

    // values[f][a][t]
    std::vector<std::vector<std::vector<std::complex<double>>>> values(fragment_count);
    std::uint64_t actual_peak_bytes = 0;
    std::uint64_t distinct_evaluations = 0;

    for (std::size_t f = 0; f < fragment_count; ++f) {
        const analysis::CutFragment& fragment = cut_plan.fragments[f];
        const std::vector<analysis::FragmentAssignment> assignments =
            analysis::enumerate_assignments(fragment);
        values[f].reserve(assignments.size());
        distinct_evaluations += assignments.size();

        for (const analysis::FragmentAssignment& assignment : assignments) {
            const Circuit fragment_circuit = analysis::build_fragment_circuit(fragment, assignment);
            Observable fragment_observable;
            fragment_observable.terms.reserve(term_count);
            for (const auto& [term, coefficient] : observable.terms) {
                static_cast<void>(coefficient);  // applied after reconstruction
                fragment_observable.terms.emplace_back(
                    analysis::build_fragment_term(fragment, term, assignment),
                    std::complex<double>{1.0, 0.0});
            }

            const EngineEstimateRequest estimate_request{fragment_circuit.num_qubits(),
                                                         &fragment_circuit.operations(),
                                                         /*want_statevector=*/false,
                                                         /*want_counts=*/false,
                                                         /*want_expectation=*/true,
                                                         &fragment_observable,
                                                         job_spec.accuracy_policy,
                                                         plan.estimated_state_bytes};
            const cost::EngineChoice choice =
                cost::pick_cheapest_engine(plan::available_engines(), estimate_request);
            common::ensure(choice.estimate.feasible,
                           "execute_cut: no engine is feasible for a fragment the planner "
                           "already costed as feasible -- internal invariant violated");
            const SimulationEngine& engine = *plan::available_engines()[choice.engine_index];

            const EngineExecutionRequest request{
                .num_qubits = fragment_circuit.num_qubits(),
                .operations = fragment_circuit.operations(),
                .want_statevector = false,
                .want_counts = false,
                .want_expectation = true,
                .shots = 0,
                .measure_targets = {},
                .bitstring_width = 0,
                .seed = job_spec.seed,
                .global_phase = 0.0,
                .budget_bytes = choice.estimate.estimated_peak_bytes,
                .is_sole_region = false,
                .observable = &fragment_observable,
                .accuracy_policy = job_spec.accuracy_policy,
                .monitor = monitor,
                .max_threads = job_spec.max_threads,
                .fuse_diagonal_gates = job_spec.fuse_diagonal_gates,
            };
            EngineExecutionResult result = engine.execute(request);
            actual_peak_bytes = std::max(actual_peak_bytes, result.actual_bytes);
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            values[f].push_back(std::move(*result.expectation_terms));
        }
    }

    // Reconstruct each term, then apply that term's coefficient. The global
    // phase cancels in an expectation value, so it plays no part here.
    std::complex<double> expectation_value{0.0, 0.0};
    for (std::size_t t = 0; t < term_count; ++t) {
        std::vector<std::vector<std::complex<double>>> per_fragment(fragment_count);
        for (std::size_t f = 0; f < fragment_count; ++f) {
            per_fragment[f].reserve(values[f].size());
            for (const std::vector<std::complex<double>>& per_term : values[f]) {
                per_fragment[f].push_back(per_term[t]);
            }
        }
        expectation_value +=
            observable.terms[t].second * recombine_cut_term(cut_plan, per_fragment);
    }

    const double execute_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_execute).count();

    Plan executed_plan = plan;
    // Set by plan_job for every CutAndRecombine plan; not provable locally.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    CutSummary summary = plan.cut_summary.value();
    summary.actual_peak_bytes = actual_peak_bytes;
    summary.distinct_evaluations = distinct_evaluations;
    executed_plan.cut_summary = summary;

    return Result(std::map<BitString, std::uint64_t>{},
                  std::nullopt,
                  expectation_value,
                  /*exact=*/true,
                  std::move(executed_plan),
                  actual_peak_bytes,
                  execute_seconds,
                  /*sample_seconds=*/0.0);
}

Result Runtime::execute(const Plan& plan,
                        const menos::plan::JobSpec& job_spec,
                        const ExecutionMonitor* monitor) const {
    if (plan.strategy == Strategy::CutAndRecombine) {
        return execute_cut(plan, job_spec, monitor);
    }
    if (plan.strategy == Strategy::DirectDense || plan.strategy == Strategy::DirectStabilizer ||
        plan.strategy == Strategy::DirectPauli || plan.strategy == Strategy::DirectDensityMatrix ||
        plan.strategy == Strategy::DirectTrajectory) {
        return execute_direct(plan, job_spec, monitor);
    }
    return execute_factorized(plan, job_spec, monitor);
}

}  // namespace menos::runtime
