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
#include "engines/dense/fused_block.h"
#include "engines/dense/sample.h"
#include "engines/dense/state_vector.h"
#include "ir/fuse_diagonal.h"
#include "ir/fuse_gates.h"
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

// The same sweep over a fusion plan's blocks (Phase 24 item 7, docs/adr/0028
// section 17): one apply_matrix per block, the gap between two blocks the
// safe boundary, `completed` counting blocks.
void apply_fused(StateVector& state,
                 std::span<const Operation> operations,
                 const ir::FusionPlan& plan,
                 const ExecutionMonitor* monitor,
                 std::uint32_t max_threads) {
    apply_blocks(
        plan,
        operations,
        monitor,
        [&](std::span<const std::complex<double>> matrix, std::span<const QubitId> operands) {
            apply_matrix(state.amplitudes(), matrix, operands, max_threads);
        });
}

// Everything execute() allocates, at the most it holds at once (Phase 21,
// docs/adr/0025 section 3.4): the state, plus the larger of
//   * one gate's transient: its generated matrix (a GateOp; a UnitaryOp's is
//     borrowed) and the generic kernel's per-chunk scratch, or with fusion one
//     block's (engines/dense/fused_block.h), beside the fusion plan that lives
//     through the sweep, and
//   * the outputs, built after the sweep and returned: the sampling buffers and
//     histogram (a sole region) or the region's cumulative distribution, and
//     one complex per observable term.
// With fusion the plan is built before the state is allocated, so its
// construction transient (docs/adr/0028 section 17) is a separate term, never
// stacked on the state. The statevector output costs nothing extra: the state
// is moved into it.
std::uint64_t peak_bytes_of(const EngineEstimateRequest& request, const ir::FusionPlan* plan) {
    const std::uint32_t n = request.num_qubits;
    const std::uint64_t state =
        common::heap_block_bytes(checked_mul(checked_pow2(n), sizeof(std::complex<double>)));
    const std::uint32_t threads = common::default_thread_count();

    std::uint64_t gate = 0;
    std::uint64_t plan_bytes = 0;
    std::uint64_t construction = 0;
    if (plan != nullptr) {
        gate = max_block_transient_bytes(*plan, *request.operations, threads);
        plan_bytes = plan->heap_bytes();
        construction = ir::FusionPlan::construction_peak_bytes(
            request.operations->size(), plan->block_count(), plan->block_qubits.size());
    } else {
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
    return std::max(construction,
                    checked_add(checked_add(state, plan_bytes), std::max(gate, outputs)));
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
namespace {

// The sweeps over the state that apply gates: one per operation with fusion
// off, one per non-barrier block with it on.
std::uint64_t gate_passes(const std::vector<Operation>& operations) {
    std::uint64_t passes = 0;
    for (const Operation& op : operations) {
        if (operand_count(op) != 0) {
            ++passes;
        }
    }
    return passes;
}

std::uint64_t gate_passes(const ir::FusionPlan& plan) {
    std::uint64_t passes = 0;
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        if (!plan.block_operands(b).empty()) {
            ++passes;
        }
    }
    return passes;
}

// The dense time terms (docs/adr/0028 section 20): component 0 counts the
// passes over the state (initialisation, the gate passes and the output
// passes), each with a fixed cost the work term does not see on a small
// state; component 1 the work units, amplitude updates weighted by 2^k;
// component 2 the shots sampled.
// A pass count and a work figure: distinct roles, named at both call sites.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
TimeTerms dense_time_terms(const EngineEstimateRequest& request,
                           std::uint64_t gate_pass_count,
                           std::uint64_t work) {
    double passes = 1.0 + static_cast<double>(gate_pass_count);
    passes += request.want_statevector ? 1.0 : 0.0;
    passes += request.want_counts ? 1.0 : 0.0;
    if (request.want_expectation) {
        passes += static_cast<double>(request.observable->terms.size());
    }
    TimeTerms terms;
    auto& dense = terms.counts[static_cast<std::size_t>(Engine::Dense)];
    dense[0] = passes;
    dense[1] = static_cast<double>(work);
    dense[2] = request.want_counts ? static_cast<double>(request.shots) : 0.0;
    return terms;
}

}  // namespace

CostEstimate DenseEngine::estimate(const EngineEstimateRequest& request) const {
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    try {
        if (!request.fuse_gates) {
            const std::uint64_t work = dense_work_units(request);
            CostEstimate estimate{/*feasible=*/true,
                                  peak_bytes_of(request, nullptr),
                                  work,
                                  /*infeasible_reason=*/"",
                                  {}};
            estimate.time = dense_time_terms(request, gate_passes(*request.operations), work);
            return estimate;
        }
        // The state's size first: a qubit count beyond the dense ceiling is
        // refused before the plan, whose operands must be below 64.
        static_cast<void>(checked_pow2(request.num_qubits));
        const ir::FusionPlan plan = ir::plan_fusion(*request.operations);
        const std::uint64_t work = dense_work_units(request, plan);
        CostEstimate estimate{
            /*feasible=*/true, peak_bytes_of(request, &plan), work, /*infeasible_reason=*/"", {}};
        estimate.fusion = FusionSummary{plan.fused_blocks, plan.gates_absorbed};
        estimate.time = dense_time_terms(request, gate_passes(plan), work);
        return estimate;
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

namespace {

// The output passes the request asks for, after the sweep's own work.
std::uint64_t output_work_units(const EngineEstimateRequest& request, std::uint64_t work) {
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

}  // namespace

std::uint64_t dense_work_units(const EngineEstimateRequest& request) {
    std::uint64_t work = checked_pow2(request.num_qubits);  // initialization
    for (const Operation& op : *request.operations) {
        const std::size_t k = operand_count(op);
        if (k == 0) {
            continue;  // MeasureOp
        }
        work = checked_add(work, checked_pow2(request.num_qubits + static_cast<std::uint32_t>(k)));
    }
    return output_work_units(request, work);
}

std::uint64_t dense_work_units(const EngineEstimateRequest& request, const ir::FusionPlan& plan) {
    std::uint64_t work = checked_pow2(request.num_qubits);  // initialization
    for (std::size_t b = 0; b < plan.block_count(); ++b) {
        const std::size_t k = plan.block_operands(b).size();
        if (k == 0) {
            continue;  // a barrier block (MeasureOp)
        }
        work = checked_add(work, checked_pow2(request.num_qubits + static_cast<std::uint32_t>(k)));
    }
    return output_work_units(request, work);
}

EngineExecutionResult DenseEngine::execute(const EngineExecutionRequest& request) const {
    // The fusion plan first, before the state, so its construction never
    // stacks on the state's allocation (peak_bytes_of). With the opt-in
    // diagonal fusion the plan is built over the diagonal-fused list, which
    // the estimate does not charge (docs/adr/0028 section 17.6).
    std::optional<std::vector<Operation>> diagonal_fused;
    std::span<const Operation> operations = request.operations;
    if (request.fuse_diagonal_gates) {
        diagonal_fused = ir::fuse_diagonal_gates(request.operations,
                                                 static_cast<std::uint32_t>(kMaxUnitaryQubits));
        operations = *diagonal_fused;
    }
    std::optional<ir::FusionPlan> plan;
    if (request.fuse_gates) {
        plan.emplace(ir::plan_fusion(operations));
    }
    StateVector state(static_cast<int>(request.num_qubits), request.budget_bytes);
    initialize(state, request.global_phase);
    if (plan.has_value()) {
        apply_fused(state, operations, *plan, request.monitor, request.max_threads);
    } else {
        apply_operations(state, operations, request.monitor, request.max_threads);
    }

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
        // One pass per term, each summed in index order: bit for bit what the
        // disk tier's chunked accumulation gives (docs/adr/0027 section 3.4)
        // and what the Phase 21 baseline pins (tests/memory/data). Batching
        // the Z-only terms through one Walsh-Hadamard transform was measured
        // on the planner-performance line and not kept at the merge: it sums
        // in butterfly order, so it is not bitwise either of those, and it
        // overwrote a state that was still to be returned (docs/adr/0028).
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
