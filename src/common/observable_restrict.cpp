#include "common/observable_restrict.h"

#include "common/pauli_row.h"
#include "menos/plan.h"

namespace menos::common {

// region_qubits (a region's local_to_global) and global_to_local are not
// realistically confusable at any call site (opposite directions of the
// same mapping, never passed from a common source).
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Observable restrict_observable(const Observable& observable,
                               const std::vector<QubitId>& region_qubits,
                               const std::vector<QubitId>& global_to_local) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const auto local_num_qubits = static_cast<std::uint32_t>(region_qubits.size());
    const std::size_t words = (static_cast<std::size_t>(local_num_qubits) + 63) / 64;

    Observable restricted;
    restricted.terms.reserve(observable.terms.size());
    for (const auto& [pauli, coefficient] : observable.terms) {
        PauliString local{local_num_qubits,
                          std::vector<std::uint64_t>(words, 0),
                          std::vector<std::uint64_t>(words, 0)};
        for (QubitId g = 0; g < pauli.num_qubits; ++g) {
            const QubitId local_q = global_to_local[g];
            if (local_q == kNoLocalQubit) {
                continue;
            }
            if (get_bit(pauli.x_words, g)) {
                set_bit(local.x_words, local_q, true);
            }
            if (get_bit(pauli.z_words, g)) {
                set_bit(local.z_words, local_q, true);
            }
        }
        restricted.terms.emplace_back(std::move(local), coefficient);
    }
    return restricted;
}

}  // namespace menos::common
