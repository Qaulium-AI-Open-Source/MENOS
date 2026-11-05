#include "menos/result.h"

#include "plan/enum_names.h"

#include <format>
#include <utility>

namespace menos {
namespace {

// Left-justifies a label to an 11-character column, matching the example
// report in docs/BUILD_PLAN.md Phase 1, 1.4. Everything else explain()
// needs (strategy/engine/backend names, candidate and region lines) is
// shared with Plan::explain() (src/plan/plan_impl.cpp) via direct
// delegation below, rather than a second copy of the same formatting.
std::string label(std::string_view text) {
    std::string s(text);
    constexpr std::size_t kColumnWidth = 11;
    if (s.size() < kColumnWidth) {
        s.append(kColumnWidth - s.size(), ' ');
    }
    return s;
}

}  // namespace

// actual_state_bytes/execute_seconds/sample_seconds are populated from
// three independent measurements at a single call site in
// Runtime::execute (src/runtime/runtime.cpp); there is no ambiguity
// about which value is which there.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Result::Result(std::map<BitString, std::uint64_t> counts,
               std::optional<std::vector<std::complex<double>>> statevector,
               std::optional<std::complex<double>> expectation_value,
               bool exact,
               Plan plan,
               std::uint64_t actual_state_bytes,
               double execute_seconds,
               double sample_seconds,
               std::vector<ReplanEvent> replans,
               std::optional<double> expectation_standard_error,
               std::uint64_t peak_host_bytes,
               std::uint64_t peak_device_bytes,
               std::uint64_t peak_disk_bytes)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    : counts_(std::move(counts)),
      statevector_(std::move(statevector)),
      expectation_value_(expectation_value),
      exact_(exact),
      plan_(std::move(plan)),
      actual_state_bytes_(actual_state_bytes),
      execute_seconds_(execute_seconds),
      sample_seconds_(sample_seconds),
      replans_(std::move(replans)),
      expectation_standard_error_(expectation_standard_error),
      peak_host_bytes_(peak_host_bytes),
      peak_device_bytes_(peak_device_bytes),
      peak_disk_bytes_(peak_disk_bytes) {}

const std::map<BitString, std::uint64_t>& Result::counts() const {
    return counts_;
}

const std::optional<std::vector<std::complex<double>>>& Result::statevector() const {
    return statevector_;
}

const std::optional<std::complex<double>>& Result::expectation_value() const {
    return expectation_value_;
}

const std::optional<double>& Result::expectation_standard_error() const {
    return expectation_standard_error_;
}

bool Result::exact() const {
    return exact_;
}

const Plan& Result::plan() const {
    return plan_;
}

std::uint64_t Result::actual_state_bytes() const {
    return actual_state_bytes_;
}

std::uint64_t Result::peak_host_bytes() const {
    return peak_host_bytes_;
}

std::uint64_t Result::peak_device_bytes() const {
    return peak_device_bytes_;
}

std::uint64_t Result::peak_disk_bytes() const {
    return peak_disk_bytes_;
}

double Result::execute_seconds() const {
    return execute_seconds_;
}

double Result::sample_seconds() const {
    return sample_seconds_;
}

double Result::plan_seconds() const {
    return plan_seconds_;
}

double Result::replan_seconds() const {
    double total = 0.0;
    for (const ReplanEvent& event : replans_) {
        total += event.plan_seconds;
    }
    return total;
}

const std::vector<ReplanEvent>& Result::replans() const {
    return replans_;
}

Result Result::with_replans(std::vector<ReplanEvent> replans) && {
    Result moved = std::move(*this);
    moved.replans_ = std::move(replans);
    return moved;
}

Result Result::with_plan_seconds(double plan_seconds) && {
    Result moved = std::move(*this);
    moved.plan_seconds_ = plan_seconds;
    return moved;
}

std::string Result::explain() const {
    // Plan::explain() (src/plan/plan_impl.cpp) produces everything up to
    // and including the "estimated" line, byte-identical to what a
    // plan-only preview of the same Plan would show (docs/BUILD_PLAN.md
    // Phase 7) -- this is the single place that text is built, so the
    // two can never drift apart.
    std::string out = plan_.explain();
    // The replan report. Emitted only when something was actually abandoned,
    // so a run that never crossed its budget -- which is every run that sets
    // no budget -- produces a report byte-identical to a pre-Phase-11 one.
    for (const ReplanEvent& event : replans_) {
        out += label("replan " + std::to_string(event.sequence)) +
               std::format("threshold={:.6g} s crossed at elapsed={:.6g} s replanned_in={:.6f} s\n",
                           event.threshold_seconds,
                           event.observed_seconds,
                           event.plan_seconds);
        out +=
            label("") + std::format("abandoned after {} safe boundaries on engine={} backend={}\n",
                                    event.boundaries_crossed,
                                    plan::engine_name(event.engine),
                                    plan::backend_name(event.backend));
        out += label("") + std::format("was  strategy={} predicted={:.6g} s\n",
                                       plan::strategy_name(event.superseded_plan.strategy),
                                       event.predicted_seconds);
        out += label("") + std::format("reason=\"{}\"\n", event.reason);
        for (std::size_t i = 0; i < event.regions.size(); ++i) {
            const RegionObservation& region = event.regions[i];
            std::string qubits;
            for (std::size_t q = 0; q < region.qubits.size(); ++q) {
                qubits += (q == 0 ? "" : ",") + std::to_string(region.qubits[q]);
            }
            out +=
                label("") + std::format(
                                "region {} qubits=[{}] engine={} backend={}{} predicted={:.6g} s "
                                "observed={:.6g} s {}\n",
                                i,
                                qubits,
                                plan::engine_name(region.engine),
                                plan::backend_name(region.backend),
                                region.tier == Tier::InMemory
                                    ? std::string()
                                    : std::format(" tier={}", plan::tier_name(region.tier)),
                                region.estimated_seconds,
                                region.observed_seconds,
                                region.completed ? "completed" : "running");
        }
    }
    if (!replans_.empty()) {
        out += label("") + std::format("now  strategy={}\n", plan::strategy_name(plan_.strategy));
    }
    // The measured peaks (docs/adr/0025) follow the pre-Phase-21 fields, so
    // a reader of the older fields finds them where they were. The initial
    // planning time (Phase 24, docs/adr/0028 section 16.7 Q3) sits with the
    // other wall-clock fields, before the peaks; the replanning time follows
    // it only when something was replanned, so a run that never replanned
    // carries no replan text at all (the Phase 11 invariant).
    out += label("actual") +
           std::format("state_bytes={} execute_s={:.6f} sample_s={:.6f} plan_s={:.6f}",
                       actual_state_bytes_,
                       execute_seconds_,
                       sample_seconds_,
                       plan_seconds_);
    if (!replans_.empty()) {
        out += std::format(" replan_s={:.6f}", replan_seconds());
    }
    out += std::format(
        " peak_host_bytes={} peak_device_bytes={}", peak_host_bytes_, peak_device_bytes_);
    // The disk tier's scratch file (Phase 23, docs/adr/0027), printed only
    // when there was one, so every other report is byte-identical to a
    // pre-Phase-23 one.
    if (peak_disk_bytes_ > 0) {
        out += std::format(" peak_disk_bytes={}", peak_disk_bytes_);
    }
    if (expectation_value_.has_value()) {
        out +=
            "\n" + label("expectation") +
            std::format("{:.12g}{:+.12g}i", expectation_value_->real(), expectation_value_->imag());
        // Only a statistical estimate carries this, so every exact report is
        // byte-identical to a pre-Phase-13 one (docs/adr/0017 section 5).
        if (expectation_standard_error_.has_value()) {
            out += std::format(" std_error={:.6g} (statistical estimate, not exact)",
                               *expectation_standard_error_);
        }
    }
    return out;
}

}  // namespace menos
