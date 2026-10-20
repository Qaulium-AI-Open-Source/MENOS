#include "engines/dense/dense_engine.h"

#include <bit>
#include <complex>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/cumulative_distribution.h"
#include "engines/dense/sample.h"
#include "engines/dense/state_vector.h"
#include "ir/fuse_diagonal.h"
#include "ir/gate_table.h"
#include "ir/noise.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::ResourceExceeded;

// Quantum operand count of an operation, 0 for MeasureOp (its cost is the
// separate sampling-pass term, not a per-operation work term). Exhaustive
// over Operation, like apply_operations below: a new alternative fails the
// static_assert until it is given its own rule, so it can never be silently
// skipped (docs/adr/0016 section 7).
std::size_t operand_count(const Operation& op) {
    return std::visit(
        [](const auto& o) -> std::size_t {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return o.qubits.size();
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                throw common::InternalError("dense: a noise channel reached a pure-state engine");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return 0;
            }
        },
        op);
}

// Applies every GateOp/UnitaryOp in operations to state, in order.
// MeasureOp entries are skipped (measurement outcomes are resolved
// separately from the runtime's own last-write-wins pass, not from
// iterating operations here).
//
// The gap between two operations is this engine's safe boundary (Phase 11,
// docs/adr/0015): the state vector is a complete, normalized state there and
// nothing has been handed to the caller, so a monitor that throws leaves
// StateVector's destructor to release the only allocation involved. monitor is
// null for every job without a time budget, which is the common case.
void apply_operations(StateVector& state,
                      std::span<const Operation> operations,
                      const ExecutionMonitor* monitor,
                      std::uint32_t max_threads) {
    std::uint64_t completed = 0;
    for (const Operation& operation : operations) {
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    const std::vector<std::complex<double>> matrix =
                        ir::generate_matrix(op.opcode, op.params);
                    apply_matrix(state.amplitudes(), matrix, op.qubits, max_threads);
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    apply_matrix(state.amplitudes(), op.matrix, op.qubits, max_threads);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // Unreachable: estimate() refuses a noisy region.
                    throw common::InternalError(
                        "dense: a noise channel reached a pure-state engine");
                } else {
                    static_assert(std::is_same_v<T, MeasureOp>);
                    // Outcomes are resolved by the sampling pass.
                }
            },
            operation);
        ++completed;
        if (monitor != nullptr) {
            monitor->at_safe_boundary(completed);
        }
    }
}

}  // namespace

std::string_view DenseEngine::name() const {
    return "dense";
}

Engine DenseEngine::engine_id() const {
    return Engine::Dense;
}

Backend DenseEngine::backend() const {
    return Backend::Cpu;
}

EngineCapabilities DenseEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/true, /*statevector=*/true, /*expectation=*/true};
}

// The work-unit cost of executing operations (GateOp/UnitaryOp only --
// MeasureOp contributes 0 work here, its cost is the separate sampling-
// pass term) on an m-qubit dense region, plus initialization and any
// requested output passes: 2^m (init) + sum over ops of 2^(m + operand
// count) + [2^m if want_statevector] + [2^m if want_counts] +
// [num_terms * 2^m if want_expectation, one pass per term]
// (docs/BUILD_PLAN.md Phase 3, 3.5 / Phase 5's expectation extension).
// Peak bytes is 16 * 2^m (complex128 state, no speculative multiplier,
// unaffected by want_expectation -- no extra allocation).
CostEstimate DenseEngine::estimate(const EngineEstimateRequest& request) const {
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    try {
        std::uint64_t work = checked_pow2(request.num_qubits);  // initialization
        for (const Operation& op : *request.operations) {
            const std::size_t k = operand_count(op);
            if (k == 0) {
                continue;  // MeasureOp
            }
            work =
                checked_add(work, checked_pow2(request.num_qubits + static_cast<std::uint32_t>(k)));
        }
        if (request.want_statevector) {
            work = checked_add(work, checked_pow2(request.num_qubits));
        }
        if (request.want_counts) {
            work = checked_add(work, checked_pow2(request.num_qubits));
        }
        if (request.want_expectation) {
            const std::uint64_t num_terms =
                static_cast<std::uint64_t>(request.observable->terms.size());
            work = checked_add(work, checked_mul(num_terms, checked_pow2(request.num_qubits)));
        }
        const std::uint64_t peak_bytes =
            checked_mul(checked_pow2(request.num_qubits), sizeof(std::complex<double>));
        return CostEstimate{/*feasible=*/true, peak_bytes, work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult DenseEngine::execute(const EngineExecutionRequest& request) const {
    StateVector state(static_cast<int>(request.num_qubits), request.budget_bytes);
    initialize(state, request.global_phase);
    if (request.fuse_diagonal_gates) {
        const std::vector<Operation> fused = ir::fuse_diagonal_gates(
            request.operations, static_cast<std::uint32_t>(kMaxUnitaryQubits));
        apply_operations(state, fused, request.monitor, request.max_threads);
    } else {
        apply_operations(state, request.operations, request.monitor, request.max_threads);
    }

    EngineExecutionResult result;
    result.actual_bytes = state.size_bytes();

    if (request.want_statevector) {
        const auto amps = state.amplitudes();
        result.statevector = std::vector<std::complex<double>>(amps.begin(), amps.end());
    }

    if (request.want_counts) {
        const std::span<const std::pair<QubitId, ClbitId>> targets = request.measure_targets;
        if (request.is_sole_region) {
            result.counts =
                sample(state, request.shots, request.seed, targets, request.bitstring_width);
        } else {
            result.counts_sampler =
                std::make_unique<CumulativeDistribution>(state, targets, request.bitstring_width);
        }
    }

    if (request.want_expectation) {
        // x_words/z_words have exactly one word here -- estimate() already
        // made this region infeasible for any num_qubits where that would
        // not hold (dense's own checked_pow2 ceiling is 62).
        const auto& all = request.observable->terms;
        std::vector<std::complex<double>> terms(all.size());
        // Z-only terms (a diagonal observable, e.g. every QUBO/Ising cost
        // Hamiltonian) are batched through one Walsh-Hadamard transform when
        // there are enough of them for its n passes to beat one pass per term.
        // That batch consumes the state, so it runs after every other output
        // and after the non-diagonal terms.
        std::vector<std::size_t> diagonal;
        std::vector<std::uint64_t> z_masks;
        for (std::size_t i = 0; i < all.size(); ++i) {
            if (all[i].first.x_words[0] == 0) {
                diagonal.push_back(i);
                z_masks.push_back(all[i].first.z_words[0]);
            }
        }
        const bool batch_diagonal =
            diagonal.size() >= 2 && diagonal.size() * 4 > request.num_qubits;
        for (std::size_t i = 0; i < all.size(); ++i) {
            if (batch_diagonal && all[i].first.x_words[0] == 0) {
                continue;
            }
            terms[i] = kernels::cpu::expectation_of_term(
                state.amplitudes(), all[i].first.x_words[0], all[i].first.z_words[0]);
        }
        if (batch_diagonal) {
            const std::vector<std::complex<double>> values =
                kernels::cpu::expectation_of_diagonal_terms(state.amplitudes(), z_masks);
            for (std::size_t j = 0; j < diagonal.size(); ++j) {
                terms[diagonal[j]] = values[j];
            }
        }
        result.expectation_terms = std::move(terms);
    }

    return result;
}

}  // namespace menos::engines::dense
