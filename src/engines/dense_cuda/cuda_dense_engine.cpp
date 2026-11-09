#include "engines/dense_cuda/cuda_dense_engine.h"

#include <algorithm>
#include <complex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/memory_ledger.h"
#include "engines/dense/cumulative_distribution.h"
#include "engines/dense/fused_block.h"
#include "engines/dense/sample.h"
#include "ir/fuse_gates.h"
#include "ir/gate_table.h"
#include "ir/noise.h"
#include "kernels/cuda/device_buffer.h"
#include "kernels/cuda/device_info.h"
#include "kernels/dense_kernels.h"

namespace menos::engines::dense_cuda {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::ResourceExceeded;

// Quantum operand count of an operation, 0 for MeasureOp -- identical to the
// CPU engine's rule, because the work-unit formula must be identical. Exhaustive
// over Operation, like apply_operations below (docs/adr/0016 section 7).
std::size_t operand_count(const Operation& op) {
    return std::visit(
        [](const auto& o) -> std::size_t {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                return o.qubits.size();
            } else if constexpr (std::is_same_v<T, ChannelOp>) {
                throw common::InternalError(
                    "cuda dense: a noise channel reached a pure-state engine");
            } else {
                static_assert(std::is_same_v<T, MeasureOp>);
                return 0;
            }
        },
        op);
}

// The gap between two operations is this backend's safe boundary (Phase 11,
// docs/adr/0015). kernels::cuda::apply_matrix synchronizes before returning, so
// at this point no kernel is in flight and the device state is complete. If the
// monitor throws, DeviceBuffer's destructor runs cudaFree during unwinding --
// the device allocation is released exactly as the host one is.
void apply_operations(kernels::DeviceState state,
                      std::span<const Operation> operations,
                      const ExecutionMonitor* monitor) {
    std::uint64_t completed = 0;
    for (const Operation& operation : operations) {
        std::visit(
            [&](const auto& op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    const std::vector<std::complex<double>> matrix =
                        ir::generate_matrix(op.opcode, op.params);
                    kernels::cuda::apply_matrix(state, matrix, op.qubits);
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    kernels::cuda::apply_matrix(state, op.matrix, op.qubits);
                } else if constexpr (std::is_same_v<T, ChannelOp>) {
                    // Unreachable: estimate() refuses a noisy region.
                    throw common::InternalError(
                        "cuda dense: a noise channel reached a pure-state engine");
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

// Everything execute() allocates on the host, at the most it holds at once
// (Phase 21, docs/adr/0025 section 3.4), in execute()'s order:
//   * during the sweep, one GateOp's generated matrix;
//   * then each term's partial sums, copied back and added up, beside the
//     terms already computed;
//   * then, for a statevector or counts, the one host copy of the state, which
//     becomes the statevector, with the CPU sampler's outputs beside it.
//   * with fusion (Phase 24 item 7), the fusion plan through the sweep and
//     one block's build beside it, and the plan's construction before it.
std::uint64_t host_peak_bytes(const EngineEstimateRequest& request,
                              std::uint64_t state_bytes,
                              const ir::FusionPlan* plan) {
    std::uint64_t gate = 0;
    std::uint64_t plan_bytes = 0;
    std::uint64_t construction = 0;
    if (plan != nullptr) {
        // The CUDA kernels take the matrix as an argument: no host scratch,
        // so one thread's figure is the block build alone.
        gate = engines::dense::max_block_transient_bytes(*plan, *request.operations, 1);
        plan_bytes = plan->heap_bytes();
        construction = ir::FusionPlan::construction_peak_bytes(
            request.operations->size(), plan->block_count(), plan->block_qubits.size());
    } else {
        for (const Operation& op : *request.operations) {
            if (const auto* g = std::get_if<GateOp>(&op)) {
                gate = std::max(gate,
                                common::heap_block_bytes(checked_mul(
                                    checked_pow2(static_cast<unsigned>(2 * g->qubits.size())),
                                    sizeof(std::complex<double>))));
            }
        }
    }
    std::uint64_t terms = 0;
    std::uint64_t reduction = 0;
    if (request.want_expectation) {
        terms = common::heap_block_bytes(
            checked_mul(static_cast<std::uint64_t>(request.observable->terms.size()),
                        sizeof(std::complex<double>)));
        reduction = checked_add(
            terms, common::heap_block_bytes(kernels::cuda::expectation_scratch_bytes()));
    }
    std::uint64_t copy_back = 0;
    if (request.want_statevector || request.want_counts) {
        copy_back = checked_add(terms, common::heap_block_bytes(state_bytes));
        if (request.want_counts) {
            const std::uint32_t n = request.num_qubits;
            copy_back =
                checked_add(copy_back,
                            request.is_sole_region
                                ? engines::dense::sample_bytes(request.shots,
                                                               std::min(n, request.bitstring_width),
                                                               request.bitstring_width)
                                : engines::dense::CumulativeDistribution::heap_bytes(
                                      checked_pow2(n), request.bitstring_width));
        }
    }
    return std::max(construction, checked_add(plan_bytes, std::max({gate, reduction, copy_back})));
}

}  // namespace

bool cuda_device_available() {
    // Queried and cached once by the kernels layer.
    return kernels::cuda::device_info().present;
}

std::string_view CudaDenseEngine::name() const {
    return "dense_cuda";
}

Engine CudaDenseEngine::engine_id() const {
    return Engine::Dense;
}

Backend CudaDenseEngine::backend() const {
    return Backend::Cuda;
}

EngineCapabilities CudaDenseEngine::capabilities() const {
    // The same three outputs the CPU dense engine offers. Counts are produced
    // by copying the state back and using the CPU sampler (see the header), so
    // advertising them here is accurate, not aspirational.
    return EngineCapabilities{/*counts=*/true, /*statevector=*/true, /*expectation=*/true};
}

// The work-unit formula is IDENTICAL to DenseEngine::estimate()'s, deliberately:
// work units measure algorithmic work, and this is the same algorithm. What
// differs between backends is the calibrated rate the planner divides by, plus
// the device-memory requirement and transfer terms reported here.
//
// Infeasible when no device is present, when the region needs more device
// memory than the GPU has, or when an operation exceeds the device kernel's
// operand ceiling -- each with a reason naming the actual numbers.
CostEstimate CudaDenseEngine::estimate(const EngineEstimateRequest& request) const {
    if (!cuda_device_available()) {
        return CostEstimate{/*feasible=*/false, 0, 0, "no CUDA device is available", {}};
    }
    if (const std::optional<std::string> reason = ir::first_noise_reason(*request.operations);
        reason.has_value()) {
        return CostEstimate{/*feasible=*/false, 0, 0, *reason, {}};
    }
    try {
        std::uint64_t work = checked_pow2(request.num_qubits);  // initialization
        // The fused sweep's blocks (Phase 24 item 7), each one launch of its
        // operand count, or every gate on its own.
        std::optional<ir::FusionPlan> plan;
        std::vector<std::size_t> pass_operand_counts;
        if (request.fuse_gates) {
            plan.emplace(ir::plan_fusion(*request.operations));
            pass_operand_counts.reserve(plan->block_count());
            for (std::size_t b = 0; b < plan->block_count(); ++b) {
                pass_operand_counts.push_back(plan->block_operands(b).size());
            }
        } else {
            pass_operand_counts.reserve(request.operations->size());
            for (const Operation& op : *request.operations) {
                pass_operand_counts.push_back(operand_count(op));
            }
        }
        for (const std::size_t k : pass_operand_counts) {
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

        const std::uint64_t state_bytes =
            checked_mul(checked_pow2(request.num_qubits), sizeof(std::complex<double>));

        // Device memory: the state buffer, plus the expectation reduction's
        // partial sums while a term is reduced. The gate matrix goes to
        // constant memory or a kernel argument.
        const std::uint64_t device_bytes =
            request.want_expectation
                ? checked_add(state_bytes, kernels::cuda::expectation_scratch_bytes())
                : state_bytes;

        // Transfers and launches, counted from what execute() actually does
        // rather than estimated in the abstract.
        std::uint64_t h2d_bytes = 0;
        std::uint64_t launches = 1;  // initialize
        for (const std::size_t k : pass_operand_counts) {
            if (k == 0) {
                continue;  // MeasureOp launches nothing
            }
            // One apply_matrix launch, whose 2^k x 2^k matrix goes up first.
            const std::uint64_t entries = checked_pow2(2 * static_cast<std::uint32_t>(k));
            h2d_bytes = checked_add(h2d_bytes, checked_mul(entries, sizeof(std::complex<double>)));
            launches = checked_add(launches, 1);
        }
        if (request.want_expectation) {
            launches =
                checked_add(launches, static_cast<std::uint64_t>(request.observable->terms.size()));
        }
        const std::uint64_t available = kernels::cuda::device_available_bytes();
        if (device_bytes > available) {
            return CostEstimate{/*feasible=*/false,
                                0,
                                0,
                                "CUDA dense region needs " + std::to_string(device_bytes) +
                                    " bytes of device memory but only " +
                                    std::to_string(available) + " bytes are available on this GPU",
                                {}};
        }

        const std::uint64_t host_bytes =
            host_peak_bytes(request, state_bytes, plan.has_value() ? &*plan : nullptr);

        // A statevector or counts request is what forces the big transfer back;
        // an expectation-only job returns one complex per term.
        const std::uint64_t d2h_bytes =
            (request.want_statevector || request.want_counts)
                ? state_bytes
                : (request.want_expectation
                       ? checked_mul(static_cast<std::uint64_t>(request.observable->terms.size()),
                                     sizeof(std::complex<double>))
                       : 0);

        CostEstimate estimate{/*feasible=*/true,
                              host_bytes,
                              work,
                              /*infeasible_reason=*/"",
                              DeviceCost{device_bytes, h2d_bytes, d2h_bytes, launches}};
        if (plan.has_value()) {
            estimate.fusion = FusionSummary{plan->fused_blocks, plan->gates_absorbed};
        }
        return estimate;
    } catch (const ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}};
    }
}

EngineExecutionResult CudaDenseEngine::execute(const EngineExecutionRequest& request) const {
    common::ensure(cuda_device_available(),
                   "CudaDenseEngine::execute: no CUDA device is available -- the planner must "
                   "not select this backend without one");

    const std::uint64_t num_amplitudes = common::checked_pow2(request.num_qubits);
    // Budgeted against the DEVICE's own free memory, not the job's host budget:
    // they are separate resources and this GPU has far less of the former.
    kernels::cuda::DeviceBuffer buffer(static_cast<std::size_t>(num_amplitudes),
                                       kernels::cuda::device_available_bytes());

    kernels::cuda::initialize(buffer.state(), request.global_phase);
    if (request.fuse_gates) {
        // The same plan the CPU engine and the disk tier build from the same
        // operations, and the same host-built block matrices (Phase 24 item
        // 7, docs/adr/0028 section 17): the device applies each with the
        // kernel that is bitwise the CPU one's, so CPU and CUDA stay
        // identical with fusion on. The plan is built before the device
        // state only in the host's accounting; the device buffer above is
        // the device's own.
        const ir::FusionPlan plan = ir::plan_fusion(request.operations);
        engines::dense::apply_blocks(
            plan,
            request.operations,
            request.monitor,
            [&](std::span<const std::complex<double>> matrix, std::span<const QubitId> operands) {
                kernels::cuda::apply_matrix(buffer.state(), matrix, operands);
            });
    } else {
        apply_operations(buffer.state(), request.operations, request.monitor);
    }

    EngineExecutionResult result;
    result.actual_bytes = buffer.size_bytes();

    // The expectation reduction runs on the device and returns one scalar per
    // term, so an expectation-only job never pays the 16 * 2^n transfer back.
    if (request.want_expectation) {
        std::vector<std::complex<double>> terms;
        terms.reserve(request.observable->terms.size());
        for (const auto& term : request.observable->terms) {
            terms.push_back(kernels::cuda::expectation_of_term(
                buffer.state(), term.first.x_words[0], term.first.z_words[0]));
        }
        result.expectation_terms = std::move(terms);
    }

    // A statevector or counts request is what forces the transfer back. Both
    // are served from one host copy, which becomes the result's statevector
    // when one is requested (docs/adr/0025 section 3.3): the device state is
    // copied straight into it, the CPU sampler reads it, and it is then moved,
    // never copied again.
    if (request.want_statevector || request.want_counts) {
        const std::uint64_t host_bytes =
            common::checked_mul(num_amplitudes, sizeof(std::complex<double>));
        if (host_bytes > request.budget_bytes) {
            throw ResourceExceeded("CUDA dense: the host copy needs " + std::to_string(host_bytes) +
                                   " bytes, budget is " + std::to_string(request.budget_bytes) +
                                   " bytes");
        }
        std::vector<std::complex<double>> host(static_cast<std::size_t>(num_amplitudes));
        // Charged while this engine holds it; once moved into the result it is
        // the result's.
        common::MemoryLedger::Charge host_charge(common::buffer_bytes(host));
        buffer.copy_to_host(host);

        if (request.want_counts) {
            // The CPU sampler, unchanged. This is what makes counts identical
            // to the CPU backend's for a given seed rather than merely close
            // (see the header).
            const std::span<const std::pair<QubitId, ClbitId>> targets = request.measure_targets;
            if (request.is_sole_region) {
                result.counts = engines::dense::sample(
                    host, request.shots, request.seed, targets, request.bitstring_width);
            } else {
                result.counts_sampler = std::make_unique<engines::dense::CumulativeDistribution>(
                    host, targets, request.bitstring_width);
            }
        }

        if (request.want_statevector) {
            host_charge.release();
            result.statevector = std::move(host);
        }
    }

    return result;
}

}  // namespace menos::engines::dense_cuda
