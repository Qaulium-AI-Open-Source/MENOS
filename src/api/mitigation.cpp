#include "menos/mitigation.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>
#include <variant>

#include "api/batch_impl.h"
#include "common/error.h"
#include "ir/validate.h"
#include "mitigation/readout_mitigation.h"
#include "plan/hardware_profile.h"

namespace menos {
namespace {

using MeasuredPairs = std::vector<std::pair<QubitId, ClbitId>>;

MeasuredPairs measured_pairs_or_throw(const Circuit& circuit, std::string_view what) {
    MeasuredPairs pairs = mitigation::measured_pairs(circuit);
    if (pairs.empty()) {
        throw common::InvalidCircuit(
            std::format("readout mitigation: the {} circuit has no measurement", what));
    }
    return pairs;
}

// The readout error of the surviving (last) measurement into clbit.
std::optional<ReadoutError> surviving_readout_error(const Circuit& circuit, ClbitId clbit) {
    std::optional<ReadoutError> error;
    for (const Operation& op : circuit.operations()) {
        if (const auto* measure = std::get_if<MeasureOp>(&op);
            measure != nullptr && measure->clbit == clbit) {
            error = measure->readout_error;
        }
    }
    return error;
}

void check_calibration_pair(const Circuit& target,
                            const MeasuredPairs& target_pairs,
                            const std::array<Circuit, 2>& calibration) {
    constexpr std::array<std::string_view, 2> kNames{"all-zero calibration", "all-one calibration"};
    for (std::size_t i = 0; i < calibration.size(); ++i) {
        const Circuit& c = calibration.at(i);
        if (c.num_qubits() != target.num_qubits() || c.num_clbits() != target.num_clbits()) {
            throw common::InvalidCircuit(std::format(
                "readout mitigation: the {} circuit has {} qubits and {} clbits, the target {} "
                "and {}",
                kNames.at(i),
                c.num_qubits(),
                c.num_clbits(),
                target.num_qubits(),
                target.num_clbits()));
        }
        if (measured_pairs_or_throw(c, kNames.at(i)) != target_pairs) {
            throw common::InvalidCircuit(std::format(
                "readout mitigation: the {} circuit does not measure exactly the target's "
                "surviving (qubit, clbit) pairs",
                kNames.at(i)));
        }
    }
}

std::string label(std::string_view text) {
    std::string s(text);
    constexpr std::size_t kColumnWidth = 11;
    if (s.size() < kColumnWidth) {
        s.append(kColumnWidth - s.size(), ' ');
    }
    return s;
}

}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
MitigatedResult::MitigatedResult(std::vector<ClbitId> measured_clbits,
                                 std::uint32_t num_clbits,
                                 std::vector<double> quasi_probabilities,
                                 std::vector<double> target_standard_errors,
                                 std::vector<double> standard_errors,
                                 double gamma,
                                 std::vector<ClbitCalibration> calibration,
                                 std::uint64_t calibration_shots,
                                 Result raw,
                                 std::vector<Result> calibration_results)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    : measured_clbits_(std::move(measured_clbits)),
      num_clbits_(num_clbits),
      quasi_probabilities_(std::move(quasi_probabilities)),
      target_standard_errors_(std::move(target_standard_errors)),
      standard_errors_(std::move(standard_errors)),
      gamma_(gamma),
      calibration_(std::move(calibration)),
      calibration_shots_(calibration_shots),
      raw_(std::move(raw)),
      calibration_results_(std::move(calibration_results)) {}

const std::vector<ClbitId>& MitigatedResult::measured_clbits() const {
    return measured_clbits_;
}

BitString MitigatedResult::outcome(std::size_t index) const {
    common::ensure(index < quasi_probabilities_.size(), "MitigatedResult::outcome: out of range");
    BitString bits(num_clbits_);
    for (std::size_t k = 0; k < measured_clbits_.size(); ++k) {
        bits.set(measured_clbits_[k], ((index >> k) & 1U) != 0);
    }
    return bits;
}

const std::vector<double>& MitigatedResult::quasi_probabilities() const {
    return quasi_probabilities_;
}

const std::vector<double>& MitigatedResult::standard_errors() const {
    return standard_errors_;
}

const std::vector<double>& MitigatedResult::target_standard_errors() const {
    return target_standard_errors_;
}

double MitigatedResult::gamma() const {
    return gamma_;
}

const std::vector<ClbitCalibration>& MitigatedResult::calibration() const {
    return calibration_;
}

std::uint64_t MitigatedResult::calibration_shots() const {
    return calibration_shots_;
}

const Result& MitigatedResult::raw() const {
    return raw_;
}

const std::vector<Result>& MitigatedResult::calibration_results() const {
    return calibration_results_;
}

bool MitigatedResult::exact() const {
    return false;
}

std::string MitigatedResult::explain() const {
    std::string out = raw_.explain();
    out += "\n" + label("mitigation") +
           std::format("readout (tensored inversion) exact=false outcomes={} gamma={:.6g}\n",
                       quasi_probabilities_.size(),
                       gamma_);
    std::uint64_t target_shots = 0;
    for (const auto& [outcome, count] : raw_.counts()) {
        target_shots += count;
    }
    out += label("") +
           std::format("calibration_shots={} target_shots={}\n", calibration_shots_, target_shots);
    for (const ClbitCalibration& c : calibration_) {
        out += label("") +
               std::format(
                   "clbit {} P(1|0)={:.6g} P(0|1)={:.6g}\n", c.clbit, c.p1_given_0, c.p0_given_1);
    }
    out += label("") + "errors: target shot noise exact; calibration first-order delta method";
    return out;
}

std::array<Circuit, 2> readout_calibration_circuits(const Circuit& target) {
    const MeasuredPairs pairs = measured_pairs_or_throw(target, "target");
    std::array<Circuit, 2> calibration{Circuit(target.num_qubits(), target.num_clbits()),
                                       Circuit(target.num_qubits(), target.num_clbits())};
    for (const auto& [qubit, clbit] : pairs) {
        calibration[1].append(Opcode::X, {qubit});
    }
    for (Circuit& c : calibration) {
        for (const auto& [qubit, clbit] : pairs) {
            c.append_measure(qubit, clbit, surviving_readout_error(target, clbit));
        }
    }
    return calibration;
}

MitigatedResult run_readout_mitigated(const Circuit& target,
                                      const std::optional<std::array<Circuit, 2>>& calibration,
                                      const RunOptions& options,
                                      const ReadoutMitigationOptions& mitigation) {
    ir::validate(target);
    const MeasuredPairs pairs = measured_pairs_or_throw(target, "target");
    if (options.shots == 0) {
        throw common::InvalidCircuit("readout mitigation needs shots > 0 for the target");
    }
    if (mitigation.calibration_shots == 0) {
        throw common::InvalidCircuit("readout mitigation needs calibration_shots > 0");
    }
    const std::array<Circuit, 2> circuits_pair =
        calibration.has_value() ? *calibration : readout_calibration_circuits(target);
    check_calibration_pair(target, pairs, circuits_pair);

    // Refused before any circuit runs: the dense vectors are a cost of the
    // mitigation itself, separate from (and after) each circuit's own plan.
    const std::uint64_t needed = mitigation::peak_bytes(pairs.size());
    const std::uint64_t budget = options.memory_budget_bytes.value_or(
        plan::default_memory_budget(plan::calibrated_hardware_profile()));
    if (needed > budget) {
        throw common::ResourceExceeded(std::format(
            "readout mitigation of {} measured clbits needs {} bytes for its dense vectors, over "
            "the memory budget of {} bytes",
            pairs.size(),
            needed,
            budget));
    }

    RunOptions calibration_options = options;
    calibration_options.shots = mitigation.calibration_shots;
    calibration_options.want_statevector = false;
    calibration_options.expectation = std::nullopt;
    calibration_options.expectation_precision = std::nullopt;
    const std::vector<Circuit> circuits{target, circuits_pair[0], circuits_pair[1]};
    const std::vector<RunOptions> per_circuit{options, calibration_options, calibration_options};
    std::vector<Result> results = api::run_batch_with_options(circuits, per_circuit, options.seed);

    std::vector<ClbitId> clbits;
    clbits.reserve(pairs.size());
    for (const auto& [qubit, clbit] : pairs) {
        clbits.push_back(clbit);
    }
    const std::vector<mitigation::ClbitEstimate> estimates = mitigation::estimate_assignment(
        results[1].counts(), results[2].counts(), clbits, mitigation.calibration_shots);
    const std::vector<double> distribution =
        mitigation::empirical_distribution(results[0].counts(), clbits);
    mitigation::Mitigation m =
        mitigation::mitigate(distribution, options.shots, estimates, mitigation.calibration_shots);

    std::vector<double> target_se(m.target_variance.size());
    std::vector<double> se(m.target_variance.size());
    for (std::size_t i = 0; i < se.size(); ++i) {
        // A variance that rounding pushed below zero is zero.
        target_se[i] = std::sqrt(std::max(0.0, m.target_variance[i]));
        se[i] = std::sqrt(std::max(0.0, m.target_variance[i] + m.calibration_variance[i]));
    }
    std::vector<ClbitCalibration> report;
    report.reserve(clbits.size());
    for (std::size_t k = 0; k < clbits.size(); ++k) {
        report.push_back(ClbitCalibration{clbits[k], estimates[k].e0, estimates[k].e1});
    }

    Result raw = std::move(results[0]);
    std::vector<Result> calibration_results;
    calibration_results.push_back(std::move(results[1]));
    calibration_results.push_back(std::move(results[2]));
    return MitigatedResult(std::move(clbits),
                           target.num_clbits(),
                           std::move(m.quasi_probabilities),
                           std::move(target_se),
                           std::move(se),
                           m.gamma,
                           std::move(report),
                           mitigation.calibration_shots,
                           std::move(raw),
                           std::move(calibration_results));
}

}  // namespace menos
