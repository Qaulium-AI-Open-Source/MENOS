#include "menos/plan.h"

#include <format>

#include "common/error.h"
#include "plan/enum_names.h"

namespace menos {

namespace plan {

std::string_view strategy_name(Strategy strategy) {
    switch (strategy) {
        case Strategy::DirectDense:
            return "direct_dense";
        case Strategy::DirectStabilizer:
            return "direct_stabilizer";
        case Strategy::DirectPauli:
            return "direct_pauli";
        case Strategy::DirectMPS:
            return "direct_mps";
        case Strategy::DirectTensorNetwork:
            return "direct_tensor_network";
        case Strategy::Factorized:
            return "factorized";
        case Strategy::CutAndRecombine:
            return "cut_and_recombine";
        case Strategy::DirectDensityMatrix:
            return "direct_density_matrix";
        case Strategy::DirectTrajectory:
            return "direct_trajectory";
    }
    throw common::InternalError("strategy_name: unknown Strategy");
}

std::string_view engine_name(Engine engine) {
    switch (engine) {
        case Engine::Dense:
            return "dense";
        case Engine::Stabilizer:
            return "stabilizer";
        case Engine::Pauli:
            return "pauli";
        case Engine::MPS:
            return "mps";
        case Engine::TensorNetwork:
            return "tensor_network";
        case Engine::DensityMatrix:
            return "density_matrix";
        case Engine::Trajectory:
            return "trajectory";
    }
    throw common::InternalError("engine_name: unknown Engine");
}

std::string_view backend_name(Backend backend) {
    switch (backend) {
        case Backend::Cpu:
            return "cpu";
        case Backend::Cuda:
            return "cuda";
    }
    throw common::InternalError("backend_name: unknown Backend");
}

}  // namespace plan

namespace {

using plan::backend_name;
using plan::engine_name;
using plan::strategy_name;

// Left-justifies a label to an 11-character column, matching the example
// report in docs/BUILD_PLAN.md Phase 1, 1.4.
std::string label(std::string_view text) {
    std::string s(text);
    constexpr std::size_t kColumnWidth = 11;
    if (s.size() < kColumnWidth) {
        s.append(kColumnWidth - s.size(), ' ');
    }
    return s;
}

std::string candidate_line(std::size_t index, const CandidateSummary& c, bool show_seconds) {
    std::string s = label("candidate " + std::to_string(index));
    s += std::format(
        "strategy={} feasible={} winner={}", strategy_name(c.strategy), c.feasible, c.is_winner);
    if (c.feasible) {
        s += std::format(
            " peak_bytes={} work_units={}", c.estimated_peak_bytes, c.estimated_work_units);
        // A CUDA candidate's device memory, beside its host peak_bytes
        // (docs/adr/0025 section 5), and a Factorized candidate's when some
        // of its regions run on CUDA. CPU candidates print nothing new.
        if (c.backend == Backend::Cuda || c.device.device_bytes > 0) {
            s += std::format(" device_bytes={}", c.device.device_bytes);
        }
        // Printed only for a time-constrained plan. estimated_seconds is
        // measured, so it differs between machines; including it in every
        // report would break the byte-identical output Plan::explain()
        // promises. See Plan::time_budget_seconds (docs/adr/0014). The
        // >= 0.0 guard still applies: a negative value means the planning
        // machine was never calibrated and no conversion existed.
        if (show_seconds && c.estimated_seconds >= 0.0) {
            s += std::format(" est_seconds={:.6g}", c.estimated_seconds);
        }
    } else {
        s += std::format(" reason=\"{}\"", c.infeasible_reason);
    }
    return s;
}

std::string qubit_list(const std::vector<QubitId>& qubits) {
    std::string s = "[";
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        if (i != 0) {
            s += ",";
        }
        s += std::to_string(qubits[i]);
    }
    s += "]";
    return s;
}

}  // namespace

std::string Plan::explain() const {
    std::string out = "MENOS EXECUTION REPORT\n";
    out += label("circuit") +
           std::format("qubits={} clbits={} operations={} depth={} global_phase={}\n",
                       circuit_summary.num_qubits,
                       circuit_summary.num_clbits,
                       circuit_summary.num_operations,
                       circuit_summary.depth,
                       circuit_summary.global_phase);
    if (circuit_summary.num_noise_channels > 0 || circuit_summary.num_readout_errors > 0) {
        out += label("noise") + std::format("channels={} readout_errors={}\n",
                                            circuit_summary.num_noise_channels,
                                            circuit_summary.num_readout_errors);
    }
    out += label("light_cone") + std::format("qubits {}->{} operations {}->{}\n",
                                             light_cone.original_num_qubits,
                                             light_cone.reduced_num_qubits,
                                             light_cone.original_num_operations,
                                             light_cone.reduced_num_operations);
    const bool show_seconds = time_budget_seconds.has_value();
    if (show_seconds) {
        out += label("time_budget") + std::format("{:.6g} s\n", *time_budget_seconds);
    }
    if (requested_backend.has_value()) {
        out += label("backend") + std::format("{} (requested)\n", backend_name(*requested_backend));
    }
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        out += candidate_line(i, candidates[i], show_seconds) + "\n";
    }
    out += label("strategy") +
           std::format("{} ({})\n", strategy_name(strategy), exact ? "exact" : "approximate");
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const Region& region = regions[i];
        out +=
            label("region " + std::to_string(i)) + std::format("qubits={} engine={} backend={}",
                                                               qubit_list(region.local_to_global),
                                                               engine_name(region.engine),
                                                               backend_name(region.backend));
        if (!region.engine_selection_reason.empty()) {
            out += std::format(" why=\"{}\"", region.engine_selection_reason);
        }
        if (region.mps_accumulated_error_bound.has_value()) {
            // The bound is an a-posteriori figure, built after execution
            // from this run's own actually-discarded singular-value
            // weights -- never a pre-execution guarantee that error
            // stays at or below the Controlled(tolerance) that was
            // requested, since tolerance only bounds each truncation
            // step, not their sum (CLAUDE.md's Controlled(tolerance)
            // domain invariant; docs/adr/0009). A plan-only preview
            // (before execution) never has this set.
            out += std::format(
                " accumulated_error_bound={:.6g} (a-posteriori, from this run's own discarded "
                "weights -- not a pre-execution guarantee)",
                *region.mps_accumulated_error_bound);
        }
        if (region.tensor_network_actual_peak_bytes.has_value()) {
            // The estimated peak_bytes for this same candidate is already
            // shown in its own candidate-table line above; this is the
            // actual, measured peak reached during contraction, for direct
            // comparison against that estimate (docs/BUILD_PLAN.md Phase 8's
            // estimated-versus-actual requirement). A plan-only preview
            // never has this set.
            out += std::format(" actual_peak_bytes={}", *region.tensor_network_actual_peak_bytes);
        }
        out += "\n";
    }
    if (cut_summary.has_value()) {
        out += label("cut") + std::format("cuts={} fragments={} branches={} distinct_evals={}",
                                          cut_summary->cut_count,
                                          cut_summary->fragment_count,
                                          cut_summary->branch_count,
                                          cut_summary->distinct_evaluations);
        if (cut_summary->actual_peak_bytes.has_value()) {
            out += std::format(" actual_peak_bytes={}", *cut_summary->actual_peak_bytes);
        }
        out += "\n";
    }
    // Only a trajectory plan has this line, so every other report is unchanged
    // (Phase 13, docs/adr/0017 section 5). It says how N was derived, so a
    // reader can check the precision guarantee from the report alone.
    if (trajectory_summary.has_value()) {
        out += label("trajectory") + std::format("n={}", trajectory_summary->trajectories);
        if (trajectory_summary->precision.has_value() &&
            trajectory_summary->coefficient_bound.has_value()) {
            out += std::format(" from max(shots, max(2, ceil(C^2/eps^2))) C={:.6g} eps={:.6g}",
                               *trajectory_summary->coefficient_bound,
                               *trajectory_summary->precision);
        } else {
            out += " from shots (one shot per trajectory)";
        }
        if (trajectory_summary->standard_error.has_value()) {
            out += std::format(" std_error={:.6g}", *trajectory_summary->standard_error);
        }
        out += " (statistical, not exact)\n";
    }
    out += label("estimated") +
           std::format("state_bytes={} work_units={}", estimated_state_bytes, estimated_work_units);
    // The device memory the CUDA decisions above were checked against
    // (docs/adr/0025, Q3). Absent from every plan with no CUDA in it.
    if (device_available_bytes.has_value()) {
        out += std::format(" device_available_bytes={}", *device_available_bytes);
    }
    out += "\n";
    return out;
}

}  // namespace menos
