#include "plan/planner.h"

#include <algorithm>
#include <limits>

#include <optional>
#include <string>
#include <variant>

#include "analysis/cut.h"
#include "analysis/interaction_graph.h"
#include "analysis/light_cone.h"
#include "common/error.h"
#include "common/observable_restrict.h"
#include "common/pauli_row.h"
#include "cost/cost_model.h"
#include "cost/trajectory_count.h"
#include "ir/depth.h"
#include "ir/noise.h"
#include "ir/wire_cut_basis.h"
#include "plan/available_engines.h"
#include "plan/enum_names.h"
#include "plan/execution_engines.h"
#include "plan/region_builder.h"

namespace menos::plan {
namespace {

// The original (pre-light-cone-reduction) circuit's own summary, stored
// on Plan so Plan::explain() and Result::explain() report identical
// "circuit" facts (docs/BUILD_PLAN.md Phase 7) -- computed once here,
// not by the runtime, since light-cone reduction below only ever sees
// the reduced circuit from this point on.
CircuitSummary summarize(const Circuit& circuit) {
    std::size_t channels = 0;
    std::size_t readout_errors = 0;
    for (const Operation& op : circuit.operations()) {
        if (std::holds_alternative<ChannelOp>(op)) {
            ++channels;
        } else if (const auto* measure = std::get_if<MeasureOp>(&op);
                   measure != nullptr && measure->readout_error.has_value()) {
            ++readout_errors;
        }
    }
    return CircuitSummary{
        .num_qubits = circuit.num_qubits(),
        .num_clbits = circuit.num_clbits(),
        .num_operations = circuit.operations().size(),
        .depth = ir::depth(circuit),
        .global_phase = circuit.global_phase,
        .num_noise_channels = channels,
        .num_readout_errors = readout_errors,
    };
}

// Fixed enumeration order (docs/BUILD_PLAN.md Phase 4, extended Phase
// 5, 6, 8 and 9): DirectDense, DirectStabilizer, DirectPauli, DirectMPS,
// DirectTensorNetwork, Factorized, CutAndRecombine. Also the tie-break order -- "ties go
// to the earlier candidate". DirectPauli and DirectTensorNetwork are
// only ever pushed onto `candidates` for an expectation request (their
// capabilities() can never satisfy a counts/statevector request, so
// enumerating either there would always be a wasted infeasible entry
// with no candidate-table value), unlike DirectStabilizer/DirectMPS,
// which are always enumerated so their own infeasibility reason is
// always visible; when either is skipped, every later index shifts
// down, so indices are resolved by strategy comparison below, not by a
// fixed constant.

// The qubit(s) an output request's causal past must cover
// (docs/BUILD_PLAN.md Phase 5's light-cone pass): every qubit, for a
// statevector request (the full final state is observed); every
// measured qubit, for a counts request restricted to a subset ("already
// possible since Phase 1"); the union of every observable term's
// support, for an expectation request. A job can request more than one
// output at once (e.g. counts and an expectation together), so the
// support set is the union of whichever apply.
std::vector<QubitId> compute_light_cone_support(const Circuit& circuit,
                                                bool want_statevector,
                                                bool want_counts,
                                                const std::optional<Observable>& expectation) {
    const std::uint32_t n = circuit.num_qubits();
    if (want_statevector) {
        std::vector<QubitId> all(n);
        for (QubitId q = 0; q < n; ++q) {
            all[q] = q;
        }
        return all;
    }

    std::vector<bool> in_support(n, false);
    if (want_counts) {
        for (const Operation& op : circuit.operations()) {
            if (const auto* measure = std::get_if<MeasureOp>(&op)) {
                in_support[measure->qubit] = true;
            }
        }
    }
    if (expectation.has_value()) {
        for (const auto& [pauli, coefficient] : expectation->terms) {
            for (QubitId q = 0; q < pauli.num_qubits; ++q) {
                if (common::get_bit(pauli.x_words, q) || common::get_bit(pauli.z_words, q)) {
                    in_support[q] = true;
                }
            }
        }
    }

    std::vector<QubitId> support;
    for (QubitId q = 0; q < n; ++q) {
        if (in_support[q]) {
            support.push_back(q);
        }
    }
    return support;
}

// The one engine a candidate runs on, or nullopt for a candidate that spans
// several. Only a single-engine candidate can carry a runtime observation,
// because an observation is attributed to one (engine, backend) pair
// (docs/adr/0015). Factorized and CutAndRecombine span several engines and are
// deliberately left out rather than approximated.
std::optional<Engine> single_engine_of(Strategy strategy) {
    switch (strategy) {
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
        case Strategy::DirectTrajectory:
            return Engine::Trajectory;
        case Strategy::Factorized:
        case Strategy::CutAndRecombine:
            return std::nullopt;
    }
    throw common::InternalError("single_engine_of: unknown Strategy");
}

}  // namespace

Plan plan_job(const JobSpec& job_spec,
              const HardwareProfile& hardware_profile,
              const RuntimeObservations& observations) {
    const std::uint64_t budget = job_spec.memory_budget_bytes.value_or(static_cast<std::uint64_t>(
        static_cast<double>(hardware_profile.total_memory_bytes) * kDefaultMemoryBudgetFraction));
    const bool want_counts = job_spec.shots > 0;
    const bool want_expectation = job_spec.expectation.has_value();

    // Light-cone reduction: runs once per job, unconditionally, before
    // interaction-graph analysis (docs/BUILD_PLAN.md Phase 5). Support
    // covers the whole circuit for a plain statevector request, so the
    // reduced circuit equals the original one operation-for-operation
    // in that case -- a lossless no-op, not a special case.
    const std::vector<QubitId> support = compute_light_cone_support(
        job_spec.circuit, job_spec.want_statevector, want_counts, job_spec.expectation);
    const analysis::LightConeResult light_cone_result =
        analysis::compute_light_cone(job_spec.circuit, support);
    const Circuit& circuit = light_cone_result.reduced_circuit;

    // The job's own observable is defined over the ORIGINAL circuit's
    // qubits; every candidate and region below works on the reduced
    // circuit's qubits, so the observable is remapped once, up front,
    // via the light-cone pass's own reduced_to_original permutation
    // (a term whose support was entirely in-cone keeps every bit it
    // had; light-cone reduction never drops a qubit any term needs, by
    // construction of compute_light_cone_support above).
    std::optional<Observable> remapped_expectation;
    if (want_expectation) {
        std::vector<QubitId> original_to_reduced(job_spec.circuit.num_qubits(), kNoLocalQubit);
        for (std::size_t i = 0; i < light_cone_result.reduced_to_original.size(); ++i) {
            original_to_reduced[light_cone_result.reduced_to_original[i]] = static_cast<QubitId>(i);
        }
        remapped_expectation = common::restrict_observable(
            *job_spec.expectation, light_cone_result.reduced_to_original, original_to_reduced);
        common::ensure(remapped_expectation->terms.size() == job_spec.expectation->terms.size(),
                       "plan_job: light-cone reduction must never drop an observable term");
    }
    const Observable* reduced_observable = want_expectation ? &*remapped_expectation : nullptr;

    const std::vector<const SimulationEngine*>& engines = available_engines();
    common::ensure(engines.size() == 5,
                   "plan_job: available_engines() must be {dense, stabilizer, pauli, mps, "
                   "tensor_network} -- internal invariant violated");
    const SimulationEngine& dense_engine = *engines[0];
    const SimulationEngine& stabilizer_engine = *engines[1];
    const SimulationEngine& pauli_engine = *engines[2];
    const SimulationEngine& mps_engine = *engines[3];
    const SimulationEngine& tensor_network_engine = *engines[4];

    // Noise is judged on the REDUCED circuit: a channel the light cone dropped
    // cannot affect any requested output (every channel is trace preserving),
    // so a job whose noise is all outside the cone is planned as the noiseless
    // job it is equivalent to. Only a noisy job sees the density-matrix engine
    // (docs/adr/0016 section 5); a noiseless job's planning is unchanged.
    const bool noisy = ir::first_noise_reason(circuit.operations()).has_value();
    const std::vector<const SimulationEngine*>& factorized_engines =
        noisy ? noisy_job_engines() : engines;

    // Phase 10: work units and peak bytes become seconds through the
    // machine's calibrated throughput and bandwidth, and a time budget is
    // enforced against that. A zero in either figure means the profile was
    // never calibrated (a hand-built HardwareProfile in a unit test, for
    // instance), in which case no seconds figure exists and a job that carries
    // a time budget is refused rather than waved through -- dividing by zero
    // would otherwise make every candidate look instant.
    const double work_units_per_second = hardware_profile.work_units_per_second;
    const double bandwidth_bytes_per_second = hardware_profile.memory_bandwidth_bytes_per_second;
    const bool calibrated = work_units_per_second > 0.0 && bandwidth_bytes_per_second > 0.0;
    common::ensure(calibrated || !job_spec.time_budget_seconds.has_value(),
                   "plan_job: a time_budget_seconds was given but this HardwareProfile carries "
                   "no calibrated throughput or bandwidth, so no candidate's runtime can be "
                   "estimated -- call plan::calibrated_hardware_profile() for a calibrated "
                   "profile");

    // Wall-clock seconds for running a candidate on the CPU.
    //
    // Two independent lower bounds, combined by max().
    //
    // The work term is the cost model's own unit converted through the
    // calibrated rate. That rate is measured on a cache-resident circuit
    // (plan/hardware_profile.cpp), so on its own it is optimistic for large n.
    //
    // The traffic term is the physical floor that corrects it: a candidate that
    // materializes estimated_peak_bytes must move at least that many bytes
    // across the memory bus, so it cannot finish sooner than those bytes take
    // at the machine's measured streaming bandwidth. This bounds bytes actually
    // written once -- deliberately the weakest claim that is unconditionally
    // true for every engine, rather than an engine-specific traffic model this
    // phase has no measurement to justify (docs/adr/0014).
    const auto cpu_seconds = [&](std::uint64_t work_units, std::uint64_t peak_bytes) {
        const double work_seconds = static_cast<double>(work_units) / work_units_per_second;
        const double traffic_seconds = static_cast<double>(peak_bytes) / bandwidth_bytes_per_second;
        return std::max(work_seconds, traffic_seconds);
    };

    // Wall-clock seconds for running a candidate on the CUDA device, end to
    // end. Every term is a separately measured rate, and the launch term is
    // what correctly makes a small circuit slower here than on the CPU -- a
    // model with only the work term would claim the GPU always wins, which is
    // precisely the claim the Phase 10 gate warns against.
    const bool cuda_calibrated = hardware_profile.cuda_present &&
                                 hardware_profile.cuda_device_work_units_per_second > 0.0 &&
                                 hardware_profile.cuda_host_to_device_bytes_per_second > 0.0 &&
                                 hardware_profile.cuda_device_to_host_bytes_per_second > 0.0;
    // A forced CUDA backend that cannot be honoured at all is refused here,
    // before anything is costed (Phase 17, docs/adr/0021 section 2, owner Q1):
    // never a silent fallback to the CPU. One that can run but does not fit is
    // refused later, as ResourceExceeded, by the ordinary feasibility rules.
    if (job_spec.backend == Backend::Cuda) {
        if (find_execution_engine(Engine::Dense, Backend::Cuda) == nullptr) {
            throw common::UnsupportedFeature(
                "backend=cuda was requested, but this build of MENOS has no CUDA support");
        }
        if (!hardware_profile.cuda_present) {
            throw common::UnsupportedFeature(
                "backend=cuda was requested, but no usable CUDA device is available");
        }
        if (ir::first_noise_reason(job_spec.circuit.operations()).has_value()) {
            throw common::UnsupportedFeature(
                "backend=cuda was requested, but the circuit is noisy and the CUDA backend runs "
                "noiseless dense simulation only");
        }
    }
    const auto cuda_seconds = [&](std::uint64_t work_units, const DeviceCost& device) {
        return static_cast<double>(device.kernel_launches) *
                   hardware_profile.cuda_kernel_launch_seconds +
               static_cast<double>(device.host_to_device_bytes) /
                   hardware_profile.cuda_host_to_device_bytes_per_second +
               static_cast<double>(work_units) /
                   hardware_profile.cuda_device_work_units_per_second +
               static_cast<double>(device.device_to_host_bytes) /
                   hardware_profile.cuda_device_to_host_bytes_per_second;
    };

    // The ONE place the rule "an observation can only raise a figure" lives
    // (Phase 11, docs/adr/0015). An abandoned attempt at this same job measured
    // how long an engine actually took on a backend before it was stopped. That
    // is a fact, not a model output, so it overrides the model wherever the model
    // is the more optimistic of the two: max(), never replacement, so a bad
    // observation cannot make the planner optimistic, and the model still
    // governs wherever it is already the pessimistic one.
    //
    // Used at BOTH places a backend's seconds are computed -- when choosing the
    // dense candidate's backend, and when testing a candidate against the time
    // budget -- so the two can never disagree about what a backend costs.
    struct FlooredSeconds {
        double seconds;
        bool from_observation;
    };
    const auto raise_to_observed_floor =
        [&](Engine engine, Backend backend, double modelled_seconds) -> FlooredSeconds {
        const std::optional<double> floor = observations.floor_for(engine, backend);
        if (floor.has_value() && *floor > modelled_seconds) {
            return {*floor, true};
        }
        return {modelled_seconds, false};
    };

    // Per-region backends of the Factorized candidate (Phase 17 item 4b,
    // docs/adr/0021 section 10). Set below, before apply_time_budget runs, only
    // when at least one region moves to CUDA; empty otherwise, which leaves the
    // candidate and its plan exactly as before. factorized_mixed_seconds is the
    // candidate's runtime with those backends: the sum of each region's own.
    std::vector<Backend> factorized_backends;
    std::optional<double> factorized_mixed_seconds;

    // Applied to every candidate uniformly, after its own estimate returns, in
    // exactly the way the memory budget already is. A candidate is not
    // special-cased and the budget is never relaxed to let one through.
    const auto apply_time_budget = [&](CandidateSummary& candidate) {
        if (!candidate.feasible) {
            return;
        }
        bool from_observation = false;
        if (calibrated) {
            double seconds =
                candidate.backend == Backend::Cuda
                    ? cuda_seconds(candidate.estimated_work_units, candidate.device)
                    : cpu_seconds(candidate.estimated_work_units, candidate.estimated_peak_bytes);
            if (candidate.strategy == Strategy::Factorized &&
                factorized_mixed_seconds.has_value()) {
                seconds = *factorized_mixed_seconds;
            }
            // Only a single-engine candidate can carry an observation: it is
            // attributed to one (engine, backend) pair, and a Factorized or
            // CutAndRecombine candidate spans several engines.
            const std::optional<Engine> engine = single_engine_of(candidate.strategy);
            if (engine.has_value()) {
                const FlooredSeconds floored =
                    raise_to_observed_floor(*engine, candidate.backend, seconds);
                seconds = floored.seconds;
                from_observation = floored.from_observation;
            }
            candidate.estimated_seconds = seconds;
        }
        if (!job_spec.time_budget_seconds.has_value()) {
            return;
        }
        const double time_budget = *job_spec.time_budget_seconds;
        if (candidate.estimated_seconds > time_budget) {
            candidate.feasible = false;
            // Whether the binding figure was modelled or measured changes what
            // a reader should do about it, so the reason says which.
            candidate.infeasible_reason =
                (from_observation ? "observed runtime of at least " : "estimated runtime ") +
                std::to_string(candidate.estimated_seconds) +
                " s exceeds the job's time budget of " + std::to_string(time_budget) + " s";
        }
    };

    std::vector<CandidateSummary> candidates;

    // Candidate: DirectDense, always enumerated.
    const cost::CostEstimate direct_dense_estimate =
        cost::estimate_direct(circuit,
                              dense_engine,
                              job_spec.want_statevector,
                              want_counts,
                              want_expectation,
                              reduced_observable,
                              budget,
                              job_spec.accuracy_policy);
    const std::size_t dense_index = candidates.size();
    candidates.push_back(CandidateSummary{
        .strategy = Strategy::DirectDense,
        .feasible = direct_dense_estimate.feasible,
        .infeasible_reason = direct_dense_estimate.infeasible_reason,
        .estimated_peak_bytes = direct_dense_estimate.estimated_peak_bytes,
        .estimated_work_units = direct_dense_estimate.estimated_work_units,
        .is_winner = false,
    });

    // Branch-and-bound bound for the two candidates whose costing is itself an
    // expensive search (DirectTensorNetwork's min-fill elimination per
    // observable term, CutAndRecombine's per-fragment engine choice). The
    // winner is the feasible candidate with the fewest work units, ties to the
    // earlier one, so once a feasible candidate is in hand, a later candidate
    // whose running total is strictly above it can never win; its estimate is
    // stopped and explain() reports it as pruned, with the figure it reached.
    //
    // Enabled only where no later step of plan_job can change a candidate
    // after enumeration: with a time budget a cheaper-in-work candidate can
    // still be rejected on seconds (and a replan raises figures from
    // observations), and the CUDA backend step can replace the dense
    // candidate's work units. In every such job the bound stays at its
    // default and every candidate is costed in full, exactly as before.
    const bool branch_and_bound =
        !job_spec.time_budget_seconds.has_value() && job_spec.backend != Backend::Cuda &&
        !(job_spec.backend != Backend::Cpu && cuda_calibrated && calibrated);
    const auto incumbent_work_units = [&]() {
        std::uint64_t bound = std::numeric_limits<std::uint64_t>::max();
        if (branch_and_bound) {
            for (const CandidateSummary& candidate : candidates) {
                if (candidate.feasible) {
                    bound = std::min(bound, candidate.estimated_work_units);
                }
            }
        }
        return bound;
    };

    // Candidate: DirectStabilizer, always enumerated -- its own
    // estimate() reports infeasibility (naming the first non-Clifford
    // operation, or that a statevector was requested) so explain()
    // always shows why it was or wasn't picked (docs/BUILD_PLAN.md
    // Phase 4 acceptance criteria 2/3/4).
    const cost::CostEstimate direct_stabilizer_estimate =
        cost::estimate_direct(circuit,
                              stabilizer_engine,
                              job_spec.want_statevector,
                              want_counts,
                              want_expectation,
                              reduced_observable,
                              budget,
                              job_spec.accuracy_policy);
    const std::size_t stabilizer_index = candidates.size();
    candidates.push_back(CandidateSummary{
        .strategy = Strategy::DirectStabilizer,
        .feasible = direct_stabilizer_estimate.feasible,
        .infeasible_reason = direct_stabilizer_estimate.infeasible_reason,
        .estimated_peak_bytes = direct_stabilizer_estimate.estimated_peak_bytes,
        .estimated_work_units = direct_stabilizer_estimate.estimated_work_units,
        .is_winner = false,
    });

    // Candidate: DirectPauli, only enumerated for an expectation request
    // (docs/BUILD_PLAN.md Phase 5's planner order) -- its capabilities()
    // can never satisfy a counts/statevector-only request, so
    // enumerating it there would always be an infeasible entry with no
    // candidate-table value, unlike DirectStabilizer's always-relevant
    // Clifford-eligibility check.
    std::optional<cost::CostEstimate> direct_pauli_estimate;
    std::optional<std::size_t> pauli_index;
    if (want_expectation) {
        direct_pauli_estimate = cost::estimate_direct(circuit,
                                                      pauli_engine,
                                                      job_spec.want_statevector,
                                                      want_counts,
                                                      want_expectation,
                                                      reduced_observable,
                                                      budget,
                                                      job_spec.accuracy_policy);
        pauli_index = candidates.size();
        candidates.push_back(CandidateSummary{
            .strategy = Strategy::DirectPauli,
            .feasible = direct_pauli_estimate->feasible,
            .infeasible_reason = direct_pauli_estimate->infeasible_reason,
            .estimated_peak_bytes = direct_pauli_estimate->estimated_peak_bytes,
            .estimated_work_units = direct_pauli_estimate->estimated_work_units,
            .is_winner = false,
        });
    }

    // Candidate: DirectMPS, always enumerated (mirrors DirectStabilizer:
    // its own estimate() reports infeasibility -- a statevector request,
    // or the rigorous/capped bond estimate exceeding budget -- so
    // explain() always shows why it was or wasn't picked,
    // docs/BUILD_PLAN.md Phase 6).
    const cost::CostEstimate direct_mps_estimate = cost::estimate_direct(circuit,
                                                                         mps_engine,
                                                                         job_spec.want_statevector,
                                                                         want_counts,
                                                                         want_expectation,
                                                                         reduced_observable,
                                                                         budget,
                                                                         job_spec.accuracy_policy);
    const std::size_t mps_index = candidates.size();
    candidates.push_back(CandidateSummary{
        .strategy = Strategy::DirectMPS,
        .feasible = direct_mps_estimate.feasible,
        .infeasible_reason = direct_mps_estimate.infeasible_reason,
        .estimated_peak_bytes = direct_mps_estimate.estimated_peak_bytes,
        .estimated_work_units = direct_mps_estimate.estimated_work_units,
        .is_winner = false,
    });

    // Candidate: DirectTensorNetwork, only enumerated for an expectation
    // request (mirrors DirectPauli's own placement above, for the same
    // reason: its capabilities() can never satisfy a counts/statevector-only
    // request). It competes on estimated_work_units like every other
    // candidate below -- no last-resort gate, no qubit-count exclusion, no
    // special casing of any kind (docs/BUILD_PLAN.md Phase 8; MENOS is
    // cost-driven, so an engine is selected because its estimate wins, never
    // because it exists). Its own estimate() bounds its internal elimination
    // search by the job's own budget (EngineEstimateRequest::budget_bytes),
    // which is what keeps an unconditionally-enumerated combinatorial search
    // from hanging planning -- the principled alternative to gating the
    // candidate itself.
    std::optional<cost::CostEstimate> direct_tensor_network_estimate;
    std::optional<std::size_t> tensor_network_index;
    if (want_expectation) {
        direct_tensor_network_estimate = cost::estimate_direct(circuit,
                                                               tensor_network_engine,
                                                               job_spec.want_statevector,
                                                               want_counts,
                                                               want_expectation,
                                                               reduced_observable,
                                                               budget,
                                                               job_spec.accuracy_policy,
                                                               /*trajectories=*/0,
                                                               incumbent_work_units());
        tensor_network_index = candidates.size();
        candidates.push_back(CandidateSummary{
            .strategy = Strategy::DirectTensorNetwork,
            .feasible = direct_tensor_network_estimate->feasible,
            .infeasible_reason = direct_tensor_network_estimate->infeasible_reason,
            .estimated_peak_bytes = direct_tensor_network_estimate->estimated_peak_bytes,
            .estimated_work_units = direct_tensor_network_estimate->estimated_work_units,
            .is_winner = false,
        });
    }

    // Candidate: DirectDensityMatrix, only for a noisy job (Phase 12,
    // docs/adr/0016 section 5). It is the only Direct candidate that can run
    // noise; every other Direct candidate above is still listed, infeasible
    // with its own noise reason, so explain() shows why each lost.
    std::optional<cost::CostEstimate> direct_density_matrix_estimate;
    std::optional<std::size_t> density_matrix_index;
    if (noisy) {
        direct_density_matrix_estimate = cost::estimate_direct(circuit,
                                                               *noisy_job_engines().back(),
                                                               job_spec.want_statevector,
                                                               want_counts,
                                                               want_expectation,
                                                               reduced_observable,
                                                               budget,
                                                               job_spec.accuracy_policy);
        density_matrix_index = candidates.size();
        candidates.push_back(CandidateSummary{
            .strategy = Strategy::DirectDensityMatrix,
            .feasible = direct_density_matrix_estimate->feasible,
            .infeasible_reason = direct_density_matrix_estimate->infeasible_reason,
            .estimated_peak_bytes = direct_density_matrix_estimate->estimated_peak_bytes,
            .estimated_work_units = direct_density_matrix_estimate->estimated_work_units,
            .is_winner = false,
        });
    }

    // Candidate: DirectTrajectory (Phase 13, docs/adr/0017 section 5). Only for
    // a noisy job whose caller opted in, so every other job -- noiseless, or
    // noisy without an opt-in -- plans exactly as before. Each requested output
    // must be opted into on its own (owner decision, 2026-10-02): an
    // expectation by expectation_precision, counts by NoiseMethod::Auto. When
    // the candidate is listed it competes with the density matrix on cost like
    // any other; there is no rule preferring either.
    std::optional<cost::CostEstimate> direct_trajectory_estimate;
    std::optional<std::size_t> trajectory_index;
    std::optional<cost::TrajectoryCount> trajectory_count;
    const bool trajectory_opted_in =
        job_spec.expectation_precision.has_value() || job_spec.noise_method == NoiseMethod::Auto;
    if (noisy && trajectory_opted_in) {
        std::string refusal;
        if (want_expectation && !job_spec.expectation_precision.has_value()) {
            refusal =
                "trajectory sampling needs expectation_precision to fix its trajectory count "
                "for an expectation";
        } else if (want_counts && job_spec.noise_method != NoiseMethod::Auto) {
            refusal = "counts by trajectory sampling need noise_method=Auto";
        } else {
            try {
                trajectory_count = cost::trajectory_count(
                    want_counts ? job_spec.shots : 0,
                    reduced_observable,
                    want_expectation ? job_spec.expectation_precision : std::nullopt);
            } catch (const common::ResourceExceeded& e) {
                refusal = e.what();
            }
        }
        if (refusal.empty()) {
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access): set on this path.
            const std::uint64_t trajectories = trajectory_count->trajectories;
            direct_trajectory_estimate = cost::estimate_direct(circuit,
                                                               trajectory_engine(),
                                                               job_spec.want_statevector,
                                                               want_counts,
                                                               want_expectation,
                                                               reduced_observable,
                                                               budget,
                                                               job_spec.accuracy_policy,
                                                               trajectories);
        } else {
            direct_trajectory_estimate = cost::CostEstimate{/*feasible=*/false, 0, 0, refusal, {}};
        }
        trajectory_index = candidates.size();
        candidates.push_back(CandidateSummary{
            .strategy = Strategy::DirectTrajectory,
            .feasible = direct_trajectory_estimate->feasible,
            .infeasible_reason = direct_trajectory_estimate->infeasible_reason,
            .estimated_peak_bytes = direct_trajectory_estimate->estimated_peak_bytes,
            .estimated_work_units = direct_trajectory_estimate->estimated_work_units,
            .is_winner = false,
        });
    }

    // Candidate: Factorized, only when the interaction graph has at
    // least two components.
    const analysis::InteractionGraph graph = analysis::build_interaction_graph(circuit);
    std::optional<cost::CostEstimate> factorized_estimate;
    std::optional<std::size_t> factorized_index;
    if (graph.components.size() >= 2) {
        factorized_estimate = cost::estimate_factorized(circuit,
                                                        graph,
                                                        factorized_engines,
                                                        job_spec.want_statevector,
                                                        want_counts,
                                                        want_expectation,
                                                        reduced_observable,
                                                        budget,
                                                        job_spec.accuracy_policy);
        factorized_index = candidates.size();
        candidates.push_back(CandidateSummary{
            .strategy = Strategy::Factorized,
            .feasible = factorized_estimate->feasible,
            .infeasible_reason = factorized_estimate->infeasible_reason,
            .estimated_peak_bytes = factorized_estimate->estimated_peak_bytes,
            .estimated_work_units = factorized_estimate->estimated_work_units,
            .is_winner = false,
        });
    }

    // Candidate: CutAndRecombine, appended after Factorized (docs/adr/0013
    // section 8, Phase 9). Expectation requests only, and only when a cut plan
    // actually exists for this circuit. It competes on estimated_work_units
    // like every other candidate: there is no "large circuit => cut" rule, and
    // no last-resort gate. The search that finds the cut is itself bounded
    // (analysis::kMaxCuts and the candidate caps inside find_cut_plan), which
    // is the Phase 8 lesson applied -- a candidate whose estimate() runs an
    // unbounded search slows every job that will never choose it.
    std::optional<cost::CostEstimate> cut_estimate;
    std::optional<std::size_t> cut_index;
    std::optional<analysis::CutPlan> cut_plan;
    if (want_expectation) {
        cut_plan = analysis::find_cut_plan(circuit, analysis::kMaxCuts);
        if (cut_plan.has_value()) {
            cut_estimate = cost::estimate_cut(*cut_plan,
                                              *remapped_expectation,
                                              engines,
                                              budget,
                                              job_spec.accuracy_policy,
                                              incumbent_work_units());
            cut_index = candidates.size();
            candidates.push_back(CandidateSummary{
                .strategy = Strategy::CutAndRecombine,
                .feasible = cut_estimate->feasible,
                .infeasible_reason = cut_estimate->infeasible_reason,
                .estimated_peak_bytes = cut_estimate->estimated_peak_bytes,
                .estimated_work_units = cut_estimate->estimated_work_units,
                .is_winner = false,
            });
        }
    }

    // Backend selection for the dense candidate (Phase 10, docs/adr/0014).
    //
    // The owner-decided rule is two-level: the ALGORITHM is chosen by work
    // units, exactly as before and with every existing CPU plan unchanged;
    // only then is the BACKEND chosen, by calibrated wall-clock seconds.
    // Dense is the one algorithm with a second backend, so it is the one
    // candidate this step touches.
    //
    // It runs BEFORE the time-budget check on purpose. If it ran after, a job
    // whose budget only the GPU could meet would already have been rejected on
    // the CPU estimate -- the candidate's feasibility in time has to be judged
    // against the backend it would actually run on.
    //
    // Ties go to the CPU: a GPU is chosen only when it is strictly faster,
    // never on equal footing, mirroring how the planner's other tie-breaks
    // favour the simpler option.
    if (job_spec.backend == Backend::Cuda) {
        // Forced CUDA (Phase 17). The algorithm is still chosen by work units,
        // but only among candidates that can run on CUDA: the direct dense
        // candidate, moved to the device if it fits there. Every other
        // candidate is infeasible, with a reason explain() prints. Checked above
        // that a CUDA engine and device exist and the job is noiseless.
        const SimulationEngine* cuda_engine = find_execution_engine(Engine::Dense, Backend::Cuda);
        const cost::CostEstimate cuda_estimate = cost::estimate_direct(circuit,
                                                                       *cuda_engine,
                                                                       job_spec.want_statevector,
                                                                       want_counts,
                                                                       want_expectation,
                                                                       reduced_observable,
                                                                       budget,
                                                                       job_spec.accuracy_policy);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            // Factorized is decided region by region below (item 4b).
            if (i != dense_index && candidates[i].feasible &&
                candidates[i].strategy != Strategy::Factorized) {
                candidates[i].feasible = false;
                candidates[i].infeasible_reason =
                    "backend=cuda requested: " +
                    std::string(strategy_name(candidates[i].strategy)) + " has no CUDA backend";
            }
        }
        CandidateSummary& dense_candidate = candidates[dense_index];
        if (dense_candidate.feasible) {
            if (!cuda_estimate.feasible) {
                dense_candidate.feasible = false;
                dense_candidate.infeasible_reason =
                    "backend=cuda requested: " + cuda_estimate.infeasible_reason;
            } else if (cuda_estimate.device.device_bytes >
                       hardware_profile.cuda_device_available_bytes) {
                dense_candidate.feasible = false;
                dense_candidate.infeasible_reason =
                    "backend=cuda requested: the state needs " +
                    std::to_string(cuda_estimate.device.device_bytes) +
                    " bytes of device memory, more than the " +
                    std::to_string(hardware_profile.cuda_device_available_bytes) + " available";
            } else {
                dense_candidate.backend = Backend::Cuda;
                dense_candidate.device = cuda_estimate.device;
                dense_candidate.estimated_peak_bytes = cuda_estimate.estimated_peak_bytes;
                dense_candidate.estimated_work_units = cuda_estimate.estimated_work_units;
            }
        }
    } else if (job_spec.backend != Backend::Cpu && cuda_calibrated && calibrated) {
        // Backend::Cpu skips this block, so nothing is moved to the GPU.
        const SimulationEngine* cuda_engine = find_execution_engine(Engine::Dense, Backend::Cuda);
        if (cuda_engine != nullptr) {
            const cost::CostEstimate cuda_estimate =
                cost::estimate_direct(circuit,
                                      *cuda_engine,
                                      job_spec.want_statevector,
                                      want_counts,
                                      want_expectation,
                                      reduced_observable,
                                      budget,
                                      job_spec.accuracy_policy);
            CandidateSummary& dense_candidate = candidates[dense_index];
            // The device budget is enforced HERE, against the profile, exactly
            // as the host budget is enforced in cost::estimate_direct against
            // the job's budget rather than inside an engine. The engine's own
            // check is against the physical card and cannot see a profile, so
            // without this a caller-supplied device budget would be ignored.
            const bool fits_device =
                cuda_estimate.device.device_bytes <= hardware_profile.cuda_device_available_bytes;
            if (cuda_estimate.feasible && dense_candidate.feasible && fits_device) {
                // Each backend's figure is raised to what an abandoned attempt
                // actually measured on it BEFORE the two are compared (Phase 11,
                // docs/adr/0015). Without this, a GPU run that overran its budget
                // would leave the model still preferring the GPU: the comparison
                // would pick it again, and only afterwards would the time-budget
                // test find it infeasible -- refusing the job even though the CPU,
                // never having been tried, was estimated to fit. With it, an
                // observed overrun on one backend steers the choice to the other,
                // and if BOTH backends carry an over-budget floor the chosen one is
                // infeasible and the job is refused, as it should be.
                //
                // With no observations both floors are absent and this is exactly
                // the Phase 10 comparison, so every first planning is unchanged.
                const double on_cpu =
                    raise_to_observed_floor(Engine::Dense,
                                            Backend::Cpu,
                                            cpu_seconds(dense_candidate.estimated_work_units,
                                                        dense_candidate.estimated_peak_bytes))
                        .seconds;
                const double on_cuda =
                    raise_to_observed_floor(
                        Engine::Dense,
                        Backend::Cuda,
                        cuda_seconds(cuda_estimate.estimated_work_units, cuda_estimate.device))
                        .seconds;
                if (on_cuda < on_cpu) {
                    dense_candidate.backend = Backend::Cuda;
                    dense_candidate.device = cuda_estimate.device;
                    dense_candidate.estimated_peak_bytes = cuda_estimate.estimated_peak_bytes;
                    dense_candidate.estimated_work_units = cuda_estimate.estimated_work_units;
                }
            }
        }
    }

    // Per-region backends for the Factorized candidate (Phase 17 item 4b,
    // docs/adr/0021 section 10, owner decision 10.1). Its regions run one after
    // another and each allocates its own state, so each dense region is decided
    // on its own, by the same comparison as the direct dense candidate above:
    // CUDA only if it fits the device and is strictly faster (ties go to the
    // CPU). A non-dense region has no CUDA engine and stays on the CPU. No
    // observed floor applies: a Factorized candidate never carries one,
    // because elapsed time is not attributed per region (Phase 11).
    //
    // Forced CUDA (item 1): the candidate is feasible only if every non-idle
    // region is dense and fits the device, and then all of them run on CUDA.
    // Forced CPU skips this entirely. CutAndRecombine is not decided here: its
    // fragments stay on the CPU (measured, not adopted, docs/adr/0021 10.1).
    if (factorized_index.has_value() && candidates[*factorized_index].feasible &&
        job_spec.backend != Backend::Cpu) {
        const bool forced_cuda = job_spec.backend == Backend::Cuda;
        const SimulationEngine* cuda_engine = find_execution_engine(Engine::Dense, Backend::Cuda);
        const bool timed = cuda_calibrated && calibrated;
        if (cuda_engine != nullptr && (forced_cuda || timed)) {
            const std::vector<Region> regions = build_factorized_regions(circuit,
                                                                         graph,
                                                                         factorized_engines,
                                                                         job_spec.want_statevector,
                                                                         want_counts,
                                                                         want_expectation,
                                                                         reduced_observable,
                                                                         budget,
                                                                         job_spec.accuracy_policy);
            std::vector<Backend> backends(regions.size(), Backend::Cpu);
            double mixed_seconds = 0.0;
            bool any_cuda = false;
            std::string refusal;
            for (std::size_t i = 0; i < regions.size() && refusal.empty(); ++i) {
                const Region& region = regions[i];
                if (region.estimated_state_bytes == 0) {
                    continue;  // idle: allocates nothing, runs on no engine
                }
                const double on_cpu =
                    timed ? cpu_seconds(region.estimated_work_units, region.estimated_state_bytes)
                          : 0.0;
                if (region.engine != Engine::Dense) {
                    if (forced_cuda) {
                        refusal = "region " + std::to_string(i) + " runs on " +
                                  std::string(engine_name(region.engine)) +
                                  ", which has no CUDA backend";
                    }
                    mixed_seconds += on_cpu;
                    continue;
                }
                const EngineEstimateRequest request{
                    .num_qubits = static_cast<std::uint32_t>(region.local_to_global.size()),
                    .operations = &region.operations,
                    .want_statevector = job_spec.want_statevector,
                    .want_counts = want_counts,
                    .want_expectation = want_expectation,
                    .observable = region.observable.has_value() ? &*region.observable : nullptr,
                    .accuracy_policy = job_spec.accuracy_policy,
                    .budget_bytes = budget,
                };
                const CostEstimate on_device = cuda_engine->estimate(request);
                const bool fits =
                    on_device.feasible &&
                    on_device.device.device_bytes <= hardware_profile.cuda_device_available_bytes;
                const double on_cuda =
                    timed ? cuda_seconds(on_device.estimated_work_units, on_device.device) : 0.0;
                if (forced_cuda) {
                    if (!fits) {
                        refusal = "region " + std::to_string(i) + " does not fit the device";
                        continue;
                    }
                } else if (!fits || !(on_cuda < on_cpu)) {
                    mixed_seconds += on_cpu;
                    continue;
                }
                backends[i] = Backend::Cuda;
                any_cuda = true;
                mixed_seconds += on_cuda;
            }
            CandidateSummary& factorized = candidates[*factorized_index];
            if (!refusal.empty()) {
                factorized.feasible = false;
                factorized.infeasible_reason = "backend=cuda requested: " + refusal;
            } else if (any_cuda) {
                factorized_backends = std::move(backends);
                if (timed) {
                    factorized_mixed_seconds = mixed_seconds;
                }
            } else if (forced_cuda) {
                // Every region idle: nothing can run on CUDA.
                factorized.feasible = false;
                factorized.infeasible_reason =
                    "backend=cuda requested: no region of the factorized plan runs on dense";
            }
        }
    }

    for (CandidateSummary& candidate : candidates) {
        apply_time_budget(candidate);
    }

    // Selection: reject infeasible candidates; among feasible ones pick
    // lowest work units; ties go to the earlier-enumerated candidate.
    std::optional<std::size_t> winner;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].feasible) {
            continue;
        }
        if (!winner.has_value() ||
            candidates[i].estimated_work_units < candidates[*winner].estimated_work_units) {
            winner = i;
        }
    }

    if (!winner.has_value()) {
        std::string message = "plan_job: no feasible candidate";
        if (job_spec.time_budget_seconds.has_value()) {
            message += " within the time budget of " +
                       std::to_string(*job_spec.time_budget_seconds) + " s";
        }
        message += " -- ";
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (i != 0) {
                message += "; ";
            }
            message += candidates[i].infeasible_reason;
        }
        throw common::ResourceExceeded(message);
    }
    candidates[*winner].is_winner = true;

    Plan plan;
    plan.candidates = candidates;
    plan.circuit_summary = summarize(job_spec.circuit);
    plan.time_budget_seconds = job_spec.time_budget_seconds;
    plan.requested_backend = job_spec.backend;
    plan.reduced_circuit = circuit;
    plan.light_cone = LightConeSummary{
        .original_num_qubits = job_spec.circuit.num_qubits(),
        .original_num_operations = job_spec.circuit.operations().size(),
        .reduced_num_qubits = circuit.num_qubits(),
        .reduced_num_operations = circuit.operations().size(),
    };

    if (*winner == dense_index) {
        plan.strategy = Strategy::DirectDense;
        plan.exact = true;
        plan.regions.push_back(build_direct_region(circuit,
                                                   Engine::Dense,
                                                   candidates[dense_index].backend,
                                                   direct_dense_estimate,
                                                   reduced_observable));
        plan.estimated_state_bytes = direct_dense_estimate.estimated_peak_bytes;
        plan.estimated_work_units = direct_dense_estimate.estimated_work_units;
    } else if (*winner == stabilizer_index) {
        plan.strategy = Strategy::DirectStabilizer;
        plan.exact = true;
        plan.regions.push_back(build_direct_region(circuit,
                                                   Engine::Stabilizer,
                                                   Backend::Cpu,
                                                   direct_stabilizer_estimate,
                                                   reduced_observable));
        plan.estimated_state_bytes = direct_stabilizer_estimate.estimated_peak_bytes;
        plan.estimated_work_units = direct_stabilizer_estimate.estimated_work_units;
    } else if (pauli_index.has_value() && *winner == *pauli_index) {
        plan.strategy = Strategy::DirectPauli;
        plan.exact = true;
        // direct_pauli_estimate is only unset when pauli_index is also
        // unset (both are set together above), which the enclosing
        // condition already rules out. Not provable to clang-tidy's
        // local static analysis.
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        plan.regions.push_back(build_direct_region(
            circuit, Engine::Pauli, Backend::Cpu, *direct_pauli_estimate, reduced_observable));
        plan.estimated_state_bytes = direct_pauli_estimate->estimated_peak_bytes;
        plan.estimated_work_units = direct_pauli_estimate->estimated_work_units;
        // NOLINTEND(bugprone-unchecked-optional-access)
    } else if (*winner == mps_index) {
        plan.strategy = Strategy::DirectMPS;
        // Under Exact, MPS's own estimate() is feasible only when the
        // rigorous Schmidt-rank bound fits budget, and it never
        // truncates -- a plan-time guarantee. Under Controlled, the
        // plan intends to allow approximation (whether or not this
        // particular run ends up truncating anything is a runtime
        // fact, not decidable here -- CLAUDE.md's Controlled(tolerance)
        // domain invariant); Result::exact() carries the post-execution
        // truth once runtime.cpp has actually run it.
        plan.exact = job_spec.accuracy_policy.kind == AccuracyPolicyKind::Exact;
        plan.regions.push_back(build_direct_region(
            circuit, Engine::MPS, Backend::Cpu, direct_mps_estimate, reduced_observable));
        plan.estimated_state_bytes = direct_mps_estimate.estimated_peak_bytes;
        plan.estimated_work_units = direct_mps_estimate.estimated_work_units;
    } else if (tensor_network_index.has_value() && *winner == *tensor_network_index) {
        plan.strategy = Strategy::DirectTensorNetwork;
        plan.exact = true;
        // direct_tensor_network_estimate is only unset when
        // tensor_network_index is also unset (both are set together
        // above), which the enclosing condition already rules out. Not
        // provable to clang-tidy's local static analysis.
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        plan.regions.push_back(build_direct_region(circuit,
                                                   Engine::TensorNetwork,
                                                   Backend::Cpu,
                                                   *direct_tensor_network_estimate,
                                                   reduced_observable));
        plan.estimated_state_bytes = direct_tensor_network_estimate->estimated_peak_bytes;
        plan.estimated_work_units = direct_tensor_network_estimate->estimated_work_units;
        // NOLINTEND(bugprone-unchecked-optional-access)
    } else if (density_matrix_index.has_value() && *winner == *density_matrix_index) {
        plan.strategy = Strategy::DirectDensityMatrix;
        // Exact for the requested noise model: nothing is truncated or
        // sampled except the requested shots (docs/adr/0016 section 9,
        // decision 1).
        plan.exact = true;
        // direct_density_matrix_estimate and density_matrix_index are set
        // together above. Not provable to clang-tidy's local analysis.
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        plan.regions.push_back(build_direct_region(circuit,
                                                   Engine::DensityMatrix,
                                                   Backend::Cpu,
                                                   *direct_density_matrix_estimate,
                                                   reduced_observable));
        plan.estimated_state_bytes = direct_density_matrix_estimate->estimated_peak_bytes;
        plan.estimated_work_units = direct_density_matrix_estimate->estimated_work_units;
        // NOLINTEND(bugprone-unchecked-optional-access)
    } else if (trajectory_index.has_value() && *winner == *trajectory_index) {
        plan.strategy = Strategy::DirectTrajectory;
        // Statistical by construction: trajectories sample the noise, so the
        // result is an estimate with a reported standard error, never exact
        // (docs/adr/0017 section 5).
        plan.exact = false;
        // direct_trajectory_estimate and trajectory_count are set whenever the
        // candidate is feasible, which a winner is. Not provable to clang-tidy.
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        plan.regions.push_back(build_direct_region(circuit,
                                                   Engine::Trajectory,
                                                   Backend::Cpu,
                                                   *direct_trajectory_estimate,
                                                   reduced_observable));
        plan.estimated_state_bytes = direct_trajectory_estimate->estimated_peak_bytes;
        plan.estimated_work_units = direct_trajectory_estimate->estimated_work_units;
        plan.trajectory_summary = TrajectorySummary{
            .trajectories = trajectory_count->trajectories,
            .precision = want_expectation ? job_spec.expectation_precision : std::nullopt,
            .coefficient_bound = trajectory_count->coefficient_bound,
        };
        // NOLINTEND(bugprone-unchecked-optional-access)
    } else if (cut_index.has_value() && *winner == *cut_index) {
        plan.strategy = Strategy::CutAndRecombine;
        plan.exact = true;  // every branch is enumerated; nothing is sampled
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        // One region per fragment, for explain(). Each carries the job's own
        // remapped observable rather than a restriction of it: a fragment's
        // actual Pauli string depends on which branch is being evaluated, so
        // it is built per branch at execution time, not fixed here.
        for (const analysis::CutFragment& fragment : cut_plan->fragments) {
            Region region;
            region.local_to_global = fragment.local_to_global;
            region.global_to_local.assign(circuit.num_qubits(), kNoLocalQubit);
            for (std::size_t local = 0; local < fragment.local_to_global.size(); ++local) {
                region.global_to_local[fragment.local_to_global[local]] =
                    static_cast<QubitId>(local);
            }
            region.operations = fragment.operations;
            region.engine = Engine::Dense;  // refined per assignment at execution time
            region.backend = Backend::Cpu;
            region.observable = *remapped_expectation;
            region.estimated_state_bytes = cut_estimate->estimated_peak_bytes;
            region.estimated_work_units = cut_estimate->estimated_work_units;
            plan.regions.push_back(std::move(region));
        }
        std::uint64_t branches = 1;
        std::uint64_t distinct = 0;
        for (std::size_t i = 0; i < cut_plan->cuts.size(); ++i) {
            branches *= ir::kWireCutTermCount;
        }
        for (const analysis::CutFragment& fragment : cut_plan->fragments) {
            distinct += analysis::distinct_evaluation_count(fragment);
        }
        plan.cut_summary = CutSummary{
            cut_plan->cuts.size(), cut_plan->fragments.size(), branches, distinct, std::nullopt};
        plan.estimated_state_bytes = cut_estimate->estimated_peak_bytes;
        plan.estimated_work_units = cut_estimate->estimated_work_units;
        // NOLINTEND(bugprone-unchecked-optional-access)
    } else {
        common::ensure(factorized_index.has_value() && *winner == *factorized_index,
                       "plan_job: winner index does not match an enumerated candidate");
        plan.strategy = Strategy::Factorized;
        plan.regions = build_factorized_regions(circuit,
                                                graph,
                                                factorized_engines,
                                                job_spec.want_statevector,
                                                want_counts,
                                                want_expectation,
                                                reduced_observable,
                                                budget,
                                                job_spec.accuracy_policy);
        if (!factorized_backends.empty()) {
            // Same deterministic build as the decision above, so the indices
            // match; checked rather than assumed.
            common::ensure(factorized_backends.size() == plan.regions.size(),
                           "plan_job: per-region backends do not match the factorized regions");
            for (std::size_t i = 0; i < plan.regions.size(); ++i) {
                plan.regions[i].backend = factorized_backends[i];
            }
        }
        // Exact unless the policy allows approximation AND at least one
        // region actually landed on MPS (the same reasoning as the
        // DirectMPS branch above, applied per region -- a Factorized
        // plan is only as exact as its least exact region).
        plan.exact = true;
        if (job_spec.accuracy_policy.kind == AccuracyPolicyKind::Controlled) {
            for (const Region& region : plan.regions) {
                if (region.engine == Engine::MPS) {
                    plan.exact = false;
                    break;
                }
            }
        }
        // factorized_estimate is only unset when candidates.size() does
        // not include a Factorized entry, in which case winner can only
        // be dense_index/stabilizer_index/*pauli_index/mps_index -- the
        // ensure() above already rules that out. Not provable to
        // clang-tidy's local static analysis.
        // NOLINTBEGIN(bugprone-unchecked-optional-access)
        plan.estimated_state_bytes = factorized_estimate->estimated_peak_bytes;
        plan.estimated_work_units = factorized_estimate->estimated_work_units;
        // NOLINTEND(bugprone-unchecked-optional-access)
    }

    return plan;
}

}  // namespace menos::plan
