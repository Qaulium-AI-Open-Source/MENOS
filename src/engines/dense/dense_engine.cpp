#include "engines/dense/dense_engine.h"

#include <algorithm>
#include <bit>
#include <complex>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/thread_count.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/cumulative_distribution.h"
#include "engines/dense/sample.h"
#include "engines/dense/state_vector.h"
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

// Everything execute() allocates, at the most it holds at once (Phase 21,
// docs/adr/0025 section 3.4): the state, plus the larger of
//   * one gate's transient: its generated matrix (a GateOp; a UnitaryOp's is
//     borrowed) and the generic kernel's per-chunk scratch, and
//   * the outputs, built after the sweep and returned: the sampling buffers and
//     histogram (a sole region) or the region's cumulative distribution, and
//     one complex per observable term.
// The statevector output costs nothing extra: the state is moved into it.
std::uint64_t peak_bytes_of(const EngineEstimateRequest& request) {
    const std::uint32_t n = request.num_qubits;
    const std::uint64_t state =
        common::heap_block_bytes(checked_mul(checked_pow2(n), sizeof(std::complex<double>)));
    const std::uint32_t threads = common::default_thread_count();

    std::uint64_t gate = 0;
    for (const Operation& op : *request.operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                    const std::size_t k = o.qubits.size();
                    std::uint64_t bytes = apply_matrix_scratch_bytes(k, threads);
                    if constexpr (std::is_same_v<T, GateOp>) {
                        bytes = checked_add(bytes,
                                            common::heap_block_bytes(checked_mul(
                                                checked_pow2(static_cast<unsigned>(2 * k)),
                                                sizeof(std::complex<double>))));
                    }
                    gate = std::max(gate, bytes);
                }
            },
            op);
    }

    std::uint64_t outputs = 0;
    if (request.want_counts) {
        if (request.is_sole_region) {
            outputs = sample_bytes(
                request.shots, std::min(n, request.bitstring_width), request.bitstring_width);
        } else {
            outputs = CumulativeDistribution::heap_bytes(checked_pow2(n), request.bitstring_width);
        }
    }
    if (request.want_expectation) {
        outputs = checked_add(outputs,
                              common::heap_block_bytes(checked_mul(
                                  static_cast<std::uint64_t>(request.observable->terms.size()),
                                  sizeof(std::complex<double>))));
    }
    return checked_add(state, std::max(gate, outputs));
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
// Peak bytes: peak_bytes_of above.
CostEstimate DenseEngine::estimate(const EngineEstimateRequest& request) const {
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    try {
        const std::uint64_t work = dense_work_units(request);
        return CostEstimate{
            /*feasible=*/true, peak_bytes_of(request), work, /*infeasible_reason=*/"", {}};
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

std::uint64_t dense_work_units(const EngineEstimateRequest& request) {
    std::uint64_t work = checked_pow2(request.num_qubits);  // initialization
    for (const Operation& op : *request.operations) {
        const std::size_t k = operand_count(op);
        if (k == 0) {
            continue;  // MeasureOp
        }
        work = checked_add(work, checked_pow2(request.num_qubits + static_cast<std::uint32_t>(k)));
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
    return work;
}

EngineExecutionResult DenseEngine::execute(const EngineExecutionRequest& request) const {
    StateVector state(static_cast<int>(request.num_qubits), request.budget_bytes);
    initialize(state, request.global_phase);
    apply_operations(state, request.operations, request.monitor, request.max_threads);

    EngineExecutionResult result;
    result.actual_bytes = state.size_bytes();

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
        std::vector<std::complex<double>> terms;
        terms.reserve(request.observable->terms.size());
        for (const auto& term : request.observable->terms) {
            // x_words/z_words have exactly one word here -- estimate()
            // already made this region infeasible for any num_qubits where
            // that would not hold (dense's own checked_pow2 ceiling is 62).
            terms.push_back(kernels::cpu::expectation_of_term(
                state.amplitudes(), term.first.x_words[0], term.first.z_words[0]));
        }
        result.expectation_terms = std::move(terms);
    }

    // Last, after sampling and the expectation have read the state: the
    // amplitudes are moved into the result, not copied (docs/adr/0025).
    if (request.want_statevector) {
        result.statevector = std::move(state).release();
    }

    return result;
}

}  // namespace menos::engines::dense
