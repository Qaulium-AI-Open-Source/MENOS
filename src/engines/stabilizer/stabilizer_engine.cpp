#include "engines/stabilizer/stabilizer_engine.h"

#include <algorithm>
#include <complex>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/memory_ledger.h"
#include "engines/stabilizer/tableau.h"
#include "ir/gate_table.h"
#include "ir/noise.h"

namespace menos::engines::stabilizer {
namespace {

using common::checked_add;
using common::checked_mul;
using common::ResourceExceeded;

void apply_gate(Tableau& tableau, const GateOp& gate) {
    switch (gate.opcode) {
        case Opcode::X:
            tableau.apply_x(gate.qubits[0]);
            return;
        case Opcode::Y:
            tableau.apply_y(gate.qubits[0]);
            return;
        case Opcode::Z:
            tableau.apply_z(gate.qubits[0]);
            return;
        case Opcode::H:
            tableau.apply_h(gate.qubits[0]);
            return;
        case Opcode::S:
            tableau.apply_s(gate.qubits[0]);
            return;
        case Opcode::Sdg:
            tableau.apply_sdg(gate.qubits[0]);
            return;
        case Opcode::SX:
            tableau.apply_sx(gate.qubits[0]);
            return;
        case Opcode::SXdg:
            tableau.apply_sxdg(gate.qubits[0]);
            return;
        case Opcode::CX:
            tableau.apply_cx(gate.qubits[0], gate.qubits[1]);
            return;
        case Opcode::CZ:
            tableau.apply_cz(gate.qubits[0], gate.qubits[1]);
            return;
        case Opcode::SWAP:
            tableau.apply_swap(gate.qubits[0], gate.qubits[1]);
            return;
        default:
            throw common::InternalError(
                "stabilizer apply_gate: opcode is not Clifford -- should have been rejected by "
                "estimate()");
    }
}

// nullopt iff every operation is a named Clifford gate (MeasureOp is
// always allowed -- it has no effect on the tableau, resolved from the
// final tableau instead). Otherwise, the index and name of the first
// disqualifying operation, formatted for explain() (docs/BUILD_PLAN.md
// Phase 4: "the index and name of the first non-Clifford operation").
//
// Exhaustive over Operation, like the apply loop in execute(): a new
// alternative fails the static_assert until it is given its own rule, so it can
// never be silently treated as Clifford (docs/adr/0016 section 7).
std::optional<std::string> first_non_clifford_reason(const std::vector<Operation>& operations) {
    for (std::size_t i = 0; i < operations.size(); ++i) {
        std::optional<std::string> reason = std::visit(
            [i](const auto& o) -> std::optional<std::string> {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    if (!ir::gate_info(o.opcode).is_clifford) {
                        return "operation " + std::to_string(i) + " (" +
                               std::string(ir::gate_info(o.opcode).name) +
                               ") is not a Clifford gate";
                    }
                    return std::nullopt;
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    return "operation " + std::to_string(i) + " (unitary) is not a Clifford gate";
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // estimate() already refused it through ir::first_noise_reason.
                    return "operation " + std::to_string(i) +
                           " (noise channel) is not a Clifford gate";
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    return std::nullopt;
                }
            },
            operations[i]);
        if (reason.has_value()) {
            return reason;
        }
    }
    return std::nullopt;
}

BitString project(const std::vector<bool>& full,
                  std::span<const std::pair<QubitId, ClbitId>> measure_targets,
                  std::uint32_t bitstring_width) {
    BitString bs(bitstring_width);
    for (const auto& [qubit, clbit] : measure_targets) {
        if (full[qubit]) {
            bs.set(clbit, true);
        }
    }
    return bs;
}

// The affine subspace's base outcome XOR a random subset of its
// generators, each restricted to the region's measured (qubit, clbit)
// positions (docs/adr/0006's projection-and-sampling step). Random bits
// come from whole words drawn from the region's own common::Rng, never
// std:: distributions -- a single word is not always enough (up to
// num_qubits generators can exist), unlike dense's one-word-per-shot
// need, so this draws ceil(generators.size()/64) words per shot.
class StabilizerCountsSampler : public RegionCountsSampler {
public:
    StabilizerCountsSampler(BitString base, std::vector<BitString> generators)
        : base_(std::move(base)), generators_(std::move(generators)) {
        charge_.set(common::buffer_bytes(generators_) +
                    generators_.size() * common::bitstring_words_bytes(base_.num_bits()));
    }

    [[nodiscard]] BitString sample_one(common::Rng& rng) const override {
        BitString outcome = base_;
        std::uint64_t word = 0;
        int bits_left = 0;
        for (const BitString& generator : generators_) {
            if (bits_left == 0) {
                word = rng.next_word();
                bits_left = 64;
            }
            if ((word & std::uint64_t{1}) != 0) {
                // XOR, never OR: a generator flips the base outcome's bits,
                // so a base bit that is 1 becomes 0. OR left it at 1, which
                // made a qubit whose base bit was set come out 1 in every
                // shot that selected a generator flipping it (the Phase 24
                // stabilizer sampler bug, docs/adr/0028 section 18).
                outcome ^= generator;
            }
            word >>= 1;
            --bits_left;
        }
        return outcome;
    }

private:
    BitString base_;
    std::vector<BitString> generators_;
    common::MemoryLedger::Charge charge_;
};

// Everything execute() allocates, at the most it holds at once (Phase 21,
// docs/adr/0025 section 3.4): the tableau, plus the largest of
//   * reduce(), with its working copies and the subspace it returns;
//   * sampling: the subspace, its projection to the measured clbits, the
//     sampler, each shot's outcome and (a sole region) the histogram;
//   * expectation: what sampling left behind (the histogram, or the sampler
//     a Factorized region returns), one complex per term, and
//     expectation_sign()'s working state for one term.
std::uint64_t peak_bytes_of(const EngineEstimateRequest& request) {
    const std::uint32_t n = request.num_qubits;
    const std::uint32_t width = request.bitstring_width;
    std::uint64_t transient = 0;
    std::uint64_t left_by_counts = 0;
    if (request.want_counts) {
        // The projected base and up to n generators (one reserved list), each
        // with its own words, held by the sampler.
        const std::uint64_t projected =
            checked_add(common::heap_block_bytes(checked_mul(n, sizeof(BitString))),
                        checked_mul(std::uint64_t{n} + 1, common::bitstring_heap_bytes(width)));
        const std::uint64_t sampler =
            checked_add(projected, common::heap_block_bytes(sizeof(StabilizerCountsSampler)));
        const std::uint64_t histogram =
            request.is_sole_region
                ? common::counts_map_bytes(
                      common::distinct_outcomes_bound(request.shots, std::min(n, width)), width)
                : 0;
        const std::uint64_t sampling =
            checked_add(checked_add(Tableau::subspace_bytes(n), sampler),
                        checked_add(histogram, common::bitstring_heap_bytes(width)));
        transient = std::max(Tableau::reduce_peak_bytes(n), sampling);
        left_by_counts = request.is_sole_region ? histogram : sampler;
    }
    if (request.want_expectation) {
        const std::uint64_t terms = common::heap_block_bytes(
            checked_mul(static_cast<std::uint64_t>(request.observable->terms.size()),
                        sizeof(std::complex<double>)));
        transient = std::max(
            transient,
            checked_add(checked_add(left_by_counts, terms), Tableau::expectation_peak_bytes(n)));
    }
    return checked_add(Tableau::rows_bytes(n), transient);
}

}  // namespace

std::string_view StabilizerEngine::name() const {
    return "stabilizer";
}

Engine StabilizerEngine::engine_id() const {
    return Engine::Stabilizer;
}

Backend StabilizerEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities StabilizerEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/true, /*statevector=*/false, /*expectation=*/true};
}

CostEstimate StabilizerEngine::estimate(const EngineEstimateRequest& request) const {
    if (request.want_statevector) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the stabilizer engine does not track global phase -- only the dense "
                            "engine can produce a statevector",
                            {}};
    }
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    if (const std::optional<std::string> reason = first_non_clifford_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }

    const std::uint32_t num_qubits = request.num_qubits;
    try {
        const std::uint64_t words = (std::uint64_t{num_qubits} + 63) / 64;
        std::uint64_t num_gates = 0;
        for (const Operation& op : *request.operations) {
            if (std::holds_alternative<GateOp>(op)) {
                ++num_gates;
            }
        }
        std::uint64_t work = num_qubits;  // row initialization
        work = checked_add(work, checked_mul(num_gates, num_qubits));
        // The Gauss-Jordan elimination both counts sampling (reduce())
        // and expectation (expectation_sign(), docs/adr/0007) run costs
        // the same O(n^2 * words) -- both are counted per requested
        // output, since expectation_sign() re-runs its own elimination
        // independently per term (no cross-term caching yet).
        const std::uint64_t elimination =
            checked_mul(std::uint64_t{2}, checked_mul(checked_mul(num_qubits, num_qubits), words));
        if (request.want_counts) {
            const std::uint64_t sampling = checked_mul(num_qubits, words);
            work = checked_add(work, checked_add(elimination, sampling));
        }
        if (request.want_expectation) {
            const std::uint64_t num_terms =
                static_cast<std::uint64_t>(request.observable->terms.size());
            work = checked_add(work, checked_mul(num_terms, elimination));
        }
        return CostEstimate{
            /*feasible=*/true, peak_bytes_of(request), work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult StabilizerEngine::execute(const EngineExecutionRequest& request) const {
    Tableau tableau(request.num_qubits);
    const common::MemoryLedger::Charge tableau_charge(tableau.held_bytes());
    // Between two gates the tableau is a complete, valid stabilizer state, so
    // this is a safe boundary to abandon at (Phase 11, docs/adr/0015). This is
    // also the engine with the cheapest work per boundary (~1.3 us at n=200),
    // which makes it the worst case for the monitor's own cost: one 31.4 ns
    // clock read against that is an upper bound of ~2.4%, too small to resolve
    // above this machine's run-to-run noise (docs/adr/0015), and paid only by a
    // job that set a time budget.
    std::uint64_t completed = 0;
    for (const Operation& op : request.operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    apply_gate(tableau, o);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // Unreachable: estimate() refuses a noisy region.
                    throw common::InternalError(
                        "stabilizer: a noise channel reached a pure-state engine");
                } else {
                    // UnitaryOp: never reached -- estimate() already rejected it.
                    // MeasureOp: no tableau effect; outcomes come from the final
                    // tableau's reduce(), never from iterating operations.
                    static_assert(std::is_same_v<T, UnitaryOp> || std::is_same_v<T, MeasureOp>);
                }
            },
            op);
        ++completed;
        if (request.monitor != nullptr) {
            request.monitor->at_safe_boundary(completed);
        }
    }

    EngineExecutionResult result;
    const std::uint64_t words = (std::uint64_t{request.num_qubits} + 63) / 64;
    const std::uint64_t row_bytes =
        checked_mul(std::uint64_t{2}, checked_mul(words, std::uint64_t{8}));
    result.actual_bytes = checked_add(checked_mul(request.num_qubits, row_bytes),
                                      checked_mul(words, std::uint64_t{8}));

    if (request.want_counts) {
        const Tableau::AffineSubspace subspace = tableau.reduce();
        // The subspace's n-bit vectors, held while they are projected; a
        // std::vector<bool> asks for at least one byte per 8 bits.
        const common::MemoryLedger::Charge subspace_charge(
            common::buffer_bytes(subspace.generators) +
            (subspace.generators.size() + 1) * ((std::uint64_t{request.num_qubits} + 7) / 8));
        BitString base = project(subspace.base, request.measure_targets, request.bitstring_width);
        std::vector<BitString> generators;
        generators.reserve(subspace.generators.size());
        for (const std::vector<bool>& generator : subspace.generators) {
            generators.push_back(
                project(generator, request.measure_targets, request.bitstring_width));
        }
        if (request.is_sole_region) {
            common::Rng rng(request.seed);
            const StabilizerCountsSampler sampler(std::move(base), std::move(generators));
            std::map<BitString, std::uint64_t> counts;
            // Per shot, continuing the gate sweep's counter. Each shot is cheap
            // here, but a large shot count is not, and the budget has to bound
            // that phase too.
            for (std::uint64_t shot = 0; shot < request.shots; ++shot) {
                counts[sampler.sample_one(rng)] += 1;
                ++completed;
                if (request.monitor != nullptr) {
                    request.monitor->at_safe_boundary(completed);
                }
            }
            const common::MemoryLedger::Charge histogram_charge(
                common::histogram_bytes(counts.size(), request.bitstring_width));
            result.counts = std::move(counts);
        } else {
            result.counts_sampler =
                std::make_unique<StabilizerCountsSampler>(std::move(base), std::move(generators));
        }
    }

    if (request.want_expectation) {
        std::vector<std::complex<double>> terms;
        terms.reserve(request.observable->terms.size());
        for (const auto& term : request.observable->terms) {
            const std::optional<bool> sign = tableau.expectation_sign(term.first);
            // nullopt: this term anticommutes with some generator, its
            // own expectation is exactly 0.
            terms.emplace_back(sign.has_value() ? (*sign ? -1.0 : 1.0) : 0.0, 0.0);
        }
        result.expectation_terms = std::move(terms);
    }

    return result;
}

}  // namespace menos::engines::stabilizer
