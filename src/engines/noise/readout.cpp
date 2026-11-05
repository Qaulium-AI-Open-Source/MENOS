#include "engines/noise/readout.h"

#include <optional>
#include <variant>

#include "common/memory_bounds.h"

namespace menos::engines::noise {

std::vector<ReadoutTarget> readout_targets(
    std::span<const Operation> operations,
    std::span<const std::pair<QubitId, ClbitId>> measure_targets) {
    std::vector<ReadoutTarget> out;
    for (const auto& [qubit, clbit] : measure_targets) {
        std::optional<ReadoutError> error;
        for (const Operation& op : operations) {
            if (const auto* measure = std::get_if<MeasureOp>(&op);
                measure != nullptr && measure->clbit == clbit) {
                error = measure->readout_error;
            }
        }
        if (error.has_value()) {
            out.push_back(ReadoutTarget{clbit, *error});
        }
    }
    return out;
}

std::uint64_t readout_targets_bytes(std::uint64_t num_targets) {
    return common::grown_vector_bytes(num_targets, sizeof(ReadoutTarget));
}

void apply_readout(BitString& outcome, std::span<const ReadoutTarget> targets, common::Rng& rng) {
    for (const ReadoutTarget& target : targets) {
        const bool truth = outcome.test(target.clbit);
        const double flip = truth ? target.error.assignment[2] : target.error.assignment[1];
        if (common::Rng::word_to_uniform(rng.next_word()) < flip) {
            outcome.set(target.clbit, !truth);
        }
    }
}

}  // namespace menos::engines::noise
