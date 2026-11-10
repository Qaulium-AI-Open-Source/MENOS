#include "engines/mps/mps_engine.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_ledger.h"
#include "common/rng.h"
#include "engines/mps/apply_gate.h"
#include "engines/mps/expectation.h"
#include "engines/mps/memory_model.h"
#include "engines/mps/mps_state.h"
#include "engines/mps/sampling.h"
#include "ir/gate_table.h"
#include "ir/noise.h"

namespace menos::engines::mps {
namespace {

using common::checked_add;
using common::checked_mul;
using common::ResourceExceeded;

// The operand qubits of a gate or unitary, nullptr for a MeasureOp. Exhaustive
// over Operation, like the apply loop in execute(): a new alternative fails the
// static_assert until it is given its own rule (docs/adr/0016 section 7).
const std::vector<QubitId>* operand_qubits(const Operation& op) {
    return std::visit(
        [](const auto& o) -> const std::vector<QubitId>* {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return &o.qubits;
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                throw common::InternalError("mps: a noise channel reached a pure-state engine");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return nullptr;
            }
        },
        op);
}

std::uint64_t actual_bytes_of(const MPSState& state) {
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < state.num_qubits(); ++i) {
        const SiteTensor& site = state.site(i);
        const std::uint64_t elems =
            checked_mul(checked_mul(static_cast<std::uint64_t>(site.chi_left), 2),
                        static_cast<std::uint64_t>(site.chi_right));
        total = checked_add(total, checked_mul(elems, sizeof(std::complex<double>)));
    }
    return total;
}

}  // namespace

std::string_view MPSEngine::name() const {
    return "mps";
}

Engine MPSEngine::engine_id() const {
    return Engine::MPS;
}

Backend MPSEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities MPSEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/true, /*statevector=*/false, /*expectation=*/true};
}

// estimated_bond is the largest bond execute() will build, found by replaying
// the operations on bond dimensions (memory_model.h, Phase 21, docs/adr/0025
// Q7): exact under Exact, an upper bound under Controlled (capped at
// accuracy_policy.max_bond_dimension when set; tolerance alone gives no
// a-priori reduction, CLAUDE.md: never invent a threshold without a
// derivation). Until Phase 21 it was the Schmidt-rank bound, which measurement
// showed the Exact bonds exceed, because every split keeps min(m, n) singular
// values, zero ones included. peak_bytes is the replay's peak: every site at
// its allocated size, every transient buffer of every gate, split and center
// shift, the sampler's sweep and per-shot buffers and the histogram.
// work_units follows the standard TEBD/MPS complexity: each local multi-qubit
// update (a real gate, or one gather/scatter SWAP) costs O(bond^3) (thin SVD
// of a matrix with a dimension of that order); a k-qubit gate spanning
// s sites costs 2*(s - k) SWAPs (gather + scatter) plus (k - 1) splits
// for the gate itself. Counts adds one center-repositioning sweep
// (MPSCountsSampler's own one-time setup, O(num_qubits * bond^3));
// expectation adds one O(num_qubits * bond^2) sweep per term.
CostEstimate MPSEngine::estimate(const EngineEstimateRequest& request) const {
    if (request.want_statevector) {
        return CostEstimate{/*feasible=*/false,
                            0,
                            0,
                            "the MPS engine cannot produce a statevector -- reconstructing the "
                            "full state defeats the reason to use it",
                            {}};
    }
    // Before the replay, which models a pure state.
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    try {
        const MemoryModel model = model_execution(request);
        const std::uint64_t estimated_bond = model.max_bond;
        const std::uint64_t peak_bytes = model.peak_bytes;

        const std::uint64_t chi3 =
            checked_mul(checked_mul(estimated_bond, estimated_bond), estimated_bond);
        std::uint64_t work = request.num_qubits;
        for (const Operation& op : *request.operations) {
            const std::vector<QubitId>* qubits = operand_qubits(op);
            if (qubits == nullptr || qubits->empty()) {
                continue;  // MeasureOp
            }
            const std::size_t k = qubits->size();
            if (k == 1) {
                work = checked_add(work, checked_mul(estimated_bond, 4));
                continue;
            }
            const QubitId min_q = *std::min_element(qubits->begin(), qubits->end());
            const QubitId max_q = *std::max_element(qubits->begin(), qubits->end());
            const std::uint64_t span =
                static_cast<std::uint64_t>(max_q) - static_cast<std::uint64_t>(min_q) + 1;
            const std::uint64_t local_updates =
                checked_add(checked_mul(2, span - static_cast<std::uint64_t>(k)),
                            static_cast<std::uint64_t>(k) - 1);
            work = checked_add(work, checked_mul(local_updates, chi3));
        }
        if (request.want_counts) {
            work = checked_add(work, checked_mul(request.num_qubits, chi3));
        }
        if (request.want_expectation) {
            const std::uint64_t num_terms =
                static_cast<std::uint64_t>(request.observable->terms.size());
            const std::uint64_t per_term =
                checked_mul(request.num_qubits, checked_mul(estimated_bond, estimated_bond));
            work = checked_add(work, checked_mul(num_terms, per_term));
        }

        CostEstimate estimate{/*feasible=*/true, peak_bytes, work, /*infeasible_reason=*/"", {}};
        // Time terms (docs/adr/0028 section 20), counted by the replay at
        // every operation's own bonds rather than at the peak bond the work
        // units above use (which charged every gate of a circuit as its
        // widest one): component 0 the SVDs, each with a fixed cost of its
        // own (the low-entanglement Trotter rows ran 40-57x their chi^3
        // prediction, section 16.10.1); component 1 their flops and the
        // expectation contractions; component 2 the tensor elements built or
        // touched.
        auto& mps = estimate.time.counts[static_cast<std::size_t>(Engine::MPS)];
        mps[0] = model.svd_count;
        mps[1] = model.linalg_flops;
        mps[2] = model.elements;
        return estimate;
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

// The accumulated error bound reported when any truncation occurred:
// sqrt(2) * sum_i sqrt(discarded_weight_i), a rigorous (not asymptotic)
// upper bound on the Euclidean distance between the exact and the
// truncated state, derived by induction from the triangle inequality --
// see docs/adr/0009 for the full derivation. Never a pre-execution
// guarantee: built entirely from this run's own actually-discarded
// weights, after execution (CLAUDE.md's Controlled(tolerance) domain
// invariant).
EngineExecutionResult MPSEngine::execute(const EngineExecutionRequest& request) const {
    MPSState state(request.num_qubits);
    TruncationStats stats;
    // Between two gates the MPS is a complete, canonical-form state and its
    // accumulated truncation statistics are consistent with it, so this is a
    // safe boundary to abandon at (Phase 11, docs/adr/0015). This engine is the
    // reason Phase 11 exists: its predicted runtime was measured 42x to 1167x
    // below its actual runtime, the largest deviation of any engine.
    std::uint64_t completed = 0;
    for (const Operation& op : request.operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    const std::vector<std::complex<double>> matrix =
                        ir::generate_matrix(o.opcode, o.params);
                    apply_gate(state, matrix, o.qubits, request.accuracy_policy, stats);
                    state.charge_sites();
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    apply_gate(state, o.matrix, o.qubits, request.accuracy_policy, stats);
                    state.charge_sites();
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // Unreachable: estimate() refuses a noisy region.
                    throw common::InternalError("mps: a noise channel reached a pure-state engine");
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    // Outcomes come from the sampler, not from this sweep.
                }
            },
            op);
        ++completed;
        if (request.monitor != nullptr) {
            request.monitor->at_safe_boundary(completed);
        }
    }

    EngineExecutionResult result;
    result.actual_bytes = actual_bytes_of(state);

    // Before sampling, which takes the state itself rather than a copy
    // (docs/adr/0025 section 3.3) and moves its orthogonality center. The
    // expectation therefore reads the same state, in the same gauge, as
    // before Phase 21.
    if (request.want_expectation) {
        std::vector<std::complex<double>> terms;
        terms.reserve(request.observable->terms.size());
        for (const auto& [pauli, coefficient] : request.observable->terms) {
            static_cast<void>(coefficient);  // Applied by the caller (docs/adr/0007).
            terms.push_back(compute_expectation(state, pauli));
        }
        result.expectation_terms = std::move(terms);
    }

    if (request.want_counts) {
        if (request.is_sole_region) {
            const MPSCountsSampler sampler(
                std::move(state), request.measure_targets, request.bitstring_width);
            common::Rng rng(request.seed);
            std::map<BitString, std::uint64_t> counts;
            // Per shot, continuing the same counter the gate sweep used. This
            // sampler walks the whole chain for every shot, and measurement put
            // sampling at 69% of an MPS run's wall time -- monitoring only the
            // gate sweep would have left the majority of the work unwatched, so
            // a job could overrun its budget where nothing was looking.
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
            result.counts_sampler = std::make_unique<MPSCountsSampler>(
                std::move(state), request.measure_targets, request.bitstring_width);
        }
    }

    if (stats.any_truncation()) {
        double sum_sqrt_weight = 0.0;
        for (const double weight : stats.discarded_weight_per_step) {
            sum_sqrt_weight += std::sqrt(weight);
        }
        result.accumulated_error_bound = std::sqrt(2.0) * sum_sqrt_weight;
    }

    return result;
}

}  // namespace menos::engines::mps
