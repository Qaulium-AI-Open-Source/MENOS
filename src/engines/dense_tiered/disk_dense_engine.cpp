#include "engines/dense_tiered/disk_dense_engine.h"

#include <algorithm>
#include <complex>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "common/checked_math.h"
#include "common/error.h"
#include "common/memory_bounds.h"
#include "common/thread_count.h"
#include "engines/dense/apply_matrix.h"
#include "engines/dense/dense_engine.h"
#include "engines/dense/fused_block.h"
#include "engines/dense/sample.h"
#include "engines/dense_tiered/chunked_executor.h"
#include "engines/dense_tiered/file_backing_store.h"
#include "engines/dense_tiered/pass_plan.h"
#include "ir/fuse_gates.h"
#include "ir/noise.h"

namespace menos::engines::dense_tiered {
namespace {

using common::checked_add;
using common::checked_mul;
using common::checked_pow2;
using common::heap_block_bytes;

// The fields of a request that shape the executor's run, shared by estimate()
// and execute() so the plan the planner priced is the plan that runs.
struct Shape {
    std::uint32_t num_qubits;
    std::span<const Operation> operations;
    bool want_statevector;
    bool want_counts;
    bool want_expectation;
    std::uint64_t shots;
    std::uint32_t bitstring_width;
    bool is_sole_region;
    const Observable* observable;
    std::uint64_t budget_bytes;
    std::uint32_t seam_chunk_qubits;
    std::uint32_t seam_window_qubits;
    bool fuse_gates;
};

Shape shape_of(const EngineEstimateRequest& request) {
    return Shape{request.num_qubits,
                 std::span<const Operation>(*request.operations),
                 request.want_statevector,
                 request.want_counts,
                 request.want_expectation,
                 request.shots,
                 request.bitstring_width,
                 request.is_sole_region,
                 request.observable,
                 request.budget_bytes,
                 request.tier_chunk_qubits,
                 request.tier_window_qubits,
                 request.fuse_gates};
}

Shape shape_of(const EngineExecutionRequest& request) {
    return Shape{request.num_qubits,
                 request.operations,
                 request.want_statevector,
                 request.want_counts,
                 request.want_expectation,
                 request.shots,
                 request.bitstring_width,
                 request.is_sole_region,
                 request.observable,
                 request.budget_bytes,
                 request.tier_chunk_qubits,
                 request.tier_window_qubits,
                 request.fuse_gates};
}

// Everything the executor will do, and what it costs, for one shape; or why
// it cannot run (reason non-empty).
struct Layout {
    PassPlan plan;
    // The fusion plan the passes run over (Phase 24 item 7), built here so
    // that it is the same one the estimate charged; nullopt with fusion off.
    std::optional<ir::FusionPlan> fusion;
    std::uint64_t peak_bytes = 0;
    std::uint64_t link_bytes = 0;
    // The written part of link_bytes: the initial write and one write-back
    // per pass (docs/adr/0028 section 16.11).
    std::uint64_t link_write_bytes = 0;
    std::uint64_t file_bytes = 0;
    std::string reason;
};

// The largest heap one gate holds while it is applied: the generic kernel's
// scratch, a GateOp's generated matrix, and the remapped operands.
std::uint64_t gate_transient_bytes(std::span<const Operation> operations) {
    const std::uint32_t threads = common::default_thread_count();
    std::uint64_t gate = 0;
    for (const Operation& op : operations) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp> || std::is_same_v<T, UnitaryOp>) {
                    const std::size_t k = o.qubits.size();
                    std::uint64_t bytes = dense::apply_matrix_scratch_bytes(k, threads);
                    if constexpr (std::is_same_v<T, GateOp>) {
                        bytes = checked_add(
                            bytes,
                            heap_block_bytes(checked_mul(checked_pow2(static_cast<unsigned>(2 * k)),
                                                         sizeof(std::complex<double>))));
                    }
                    bytes = checked_add(bytes, heap_block_bytes(k * sizeof(std::uint32_t)));
                    gate = std::max(gate, bytes);
                }
            },
            op);
    }
    return gate;
}

// The outputs, built after the passes and returned: the sampling buffers and
// histogram, and one complex per observable term.
std::uint64_t outputs_bytes(const Shape& shape) {
    std::uint64_t outputs = 0;
    if (shape.want_counts) {
        outputs = dense::sample_bytes(
            shape.shots, std::min(shape.num_qubits, shape.bitstring_width), shape.bitstring_width);
    }
    if (shape.want_expectation) {
        outputs = checked_add(
            outputs,
            heap_block_bytes(checked_mul(static_cast<std::uint64_t>(shape.observable->terms.size()),
                                         sizeof(std::complex<double>))));
    }
    return outputs;
}

// Whether some term's X mask has a bit at or above `chunk_qubits`, so its
// chunks are read in pairs and the window must hold two.
bool needs_chunk_pair(const Shape& shape, std::uint32_t chunk_qubits) {
    if (!shape.want_expectation || chunk_qubits >= 64) {
        return false;
    }
    for (const auto& term : shape.observable->terms) {
        if ((term.first.x_words[0] >> chunk_qubits) != 0) {
            return true;
        }
    }
    return false;
}

// The pass vector's heap, bounded by one Pass per operation: the same figure
// at estimate and at execution, so both derive the same window from a budget.
std::uint64_t pass_vector_bound(const Shape& shape) {
    return heap_block_bytes(checked_mul(shape.operations.size(), sizeof(Pass)));
}

Layout lay_out(const Shape& shape) {
    Layout layout;
    if (const std::optional<std::string> reason = ir::first_noise_reason(shape.operations);
        reason.has_value()) {
        layout.reason = *reason;
        return layout;
    }
    if (shape.want_statevector) {
        layout.reason =
            "the disk tier does not return a statevector: the whole 16 * 2^n-byte state would "
            "have to be in memory";
        return layout;
    }
    if (!shape.is_sole_region) {
        layout.reason = "the disk tier runs only as a job's sole region";
        return layout;
    }
    try {
        const std::uint32_t n = shape.num_qubits;
        // The state's size first: a qubit count beyond the dense ceiling is
        // refused here, before any pass plan would shift by its global
        // qubits.
        layout.file_bytes = checked_mul(checked_pow2(n), sizeof(std::complex<double>));
        // With fusion the passes are over blocks: the plan lives through the
        // run, one block's transient (with the executor's remapped operands)
        // replaces one gate's, and the plan's construction is a peak of its
        // own before the window exists.
        std::uint64_t gate_transient = gate_transient_bytes(shape.operations);
        std::uint64_t plan_bytes = 0;
        std::uint64_t construction = 0;
        if (shape.fuse_gates) {
            layout.fusion.emplace(ir::plan_fusion(shape.operations));
            gate_transient = checked_add(
                dense::max_block_transient_bytes(
                    *layout.fusion, shape.operations, common::default_thread_count()),
                heap_block_bytes(checked_mul(kMaxUnitaryQubits, sizeof(std::uint32_t))));
            plan_bytes = layout.fusion->heap_bytes();
            construction =
                ir::FusionPlan::construction_peak_bytes(shape.operations.size(),
                                                        layout.fusion->block_count(),
                                                        layout.fusion->block_qubits.size());
        }
        const ir::FusionPlan* fusion = layout.fusion.has_value() ? &*layout.fusion : nullptr;
        const std::uint64_t fixed = checked_add(
            checked_add(std::max(gate_transient, outputs_bytes(shape)), pass_vector_bound(shape)),
            plan_bytes);
        std::optional<PassPlan> plan;
        if (shape.seam_chunk_qubits > 0 || shape.seam_window_qubits > 0) {
            // The internal seam (docs/adr/0027 section 9 Q10): the chunk, and
            // the window, as the test asks; an unset window is the smallest
            // one that fits the gates.
            const std::uint32_t chunk_qubits = std::min(shape.seam_chunk_qubits, n);
            std::uint32_t window_qubits = shape.seam_window_qubits;
            if (window_qubits == 0) {
                const std::uint32_t widest =
                    build_pass_plan(shape.operations, n, chunk_qubits, n, fusion)
                        .max_global_operands;
                window_qubits =
                    std::min(n,
                             chunk_qubits +
                                 std::max(widest, needs_chunk_pair(shape, chunk_qubits) ? 1U : 0U));
            }
            window_qubits = std::clamp(window_qubits, chunk_qubits, n);
            PassPlan fixed_plan =
                build_pass_plan(shape.operations, n, chunk_qubits, window_qubits, fusion);
            if (fixed_plan.window_too_small) {
                layout.reason =
                    "the disk tier's window of 2^" + std::to_string(window_qubits) +
                    " amplitudes holds 2^" + std::to_string(window_qubits - chunk_qubits) +
                    " chunks of 2^" + std::to_string(chunk_qubits) + ", and a gate has " +
                    std::to_string(fixed_plan.max_global_operands) + " operands above the chunk";
                return layout;
            }
            plan = std::move(fixed_plan);
        } else {
            // The largest window the budget leaves room for beside the
            // outputs, one gate's transient and the pass plan.
            std::optional<std::uint32_t> window_qubits;
            for (std::uint32_t w = 0; w <= n; ++w) {
                if (checked_add(Window::heap_bytes(w), fixed) > shape.budget_bytes) {
                    break;
                }
                window_qubits = w;
            }
            if (!window_qubits.has_value()) {
                layout.reason = "the disk tier's smallest window, " +
                                std::to_string(checked_add(Window::heap_bytes(0), fixed)) +
                                " bytes with the outputs, exceeds the budget of " +
                                std::to_string(shape.budget_bytes) + " bytes";
                return layout;
            }
            plan = choose_pass_plan(shape.operations,
                                    n,
                                    *window_qubits,
                                    needs_chunk_pair(shape, *window_qubits),
                                    fusion);
            if (!plan.has_value()) {
                const std::uint32_t widest =
                    build_pass_plan(shape.operations, n, 0, n, fusion).max_global_operands;
                layout.reason = "no chunk size fits every gate in the disk tier's window of 2^" +
                                std::to_string(*window_qubits) +
                                " amplitudes the budget allows: a gate couples 2^" +
                                std::to_string(widest) + " chunks";
                return layout;
            }
        }
        layout.peak_bytes =
            std::max(construction, checked_add(Window::heap_bytes(plan->window_qubits), fixed));
        // The link: the initial write, each pass's read and write, the
        // sampling read, and each term's read (two when its chunks pair).
        std::uint64_t file_passes = 1 + 2 * plan->pass_count();
        if (shape.want_counts) {
            file_passes += 1;
        }
        if (shape.want_expectation) {
            for (const auto& term : shape.observable->terms) {
                file_passes += (term.first.x_words[0] >> plan->chunk_qubits) != 0
                                   ? std::uint64_t{2}
                                   : std::uint64_t{1};
            }
        }
        layout.link_bytes = checked_mul(layout.file_bytes, file_passes);
        layout.link_write_bytes = checked_mul(layout.file_bytes, 1 + plan->pass_count());
        layout.plan = std::move(*plan);
    } catch (const common::ResourceExceeded& e) {
        layout.reason = e.what();
    }
    return layout;
}

}  // namespace

std::string_view DiskDenseEngine::name() const {
    return "dense_disk";
}

Engine DiskDenseEngine::engine_id() const {
    return Engine::Dense;
}

Backend DiskDenseEngine::backend() const {
    return Backend::Cpu;
}

Tier DiskDenseEngine::tier() const {
    return Tier::Disk;
}

EngineCapabilities DiskDenseEngine::capabilities() const {
    return EngineCapabilities{/*counts=*/true, /*statevector=*/false, /*expectation=*/true};
}

CostEstimate DiskDenseEngine::estimate(const EngineEstimateRequest& request) const {
    const Layout layout = lay_out(shape_of(request));
    if (!layout.reason.empty()) {
        return CostEstimate{/*feasible=*/false, 0, 0, layout.reason, {}, {}};
    }
    try {
        const std::uint64_t work = layout.fusion.has_value()
                                       ? dense::dense_work_units(request, *layout.fusion)
                                       : dense::dense_work_units(request);
        CostEstimate estimate{/*feasible=*/true,
                              layout.peak_bytes,
                              work,
                              /*infeasible_reason=*/"",
                              {},
                              TierCost{layout.plan.pass_count(),
                                       layout.plan.chunk_qubits,
                                       layout.plan.window_qubits,
                                       layout.link_bytes,
                                       layout.file_bytes,
                                       layout.link_write_bytes}};
        if (layout.fusion.has_value()) {
            estimate.fusion =
                FusionSummary{layout.fusion->fused_blocks, layout.fusion->gates_absorbed};
        }
        return estimate;
    } catch (const common::ResourceExceeded& e) {
        return CostEstimate{/*feasible=*/false, 0, 0, e.what(), {}, {}};
    }
}

EngineExecutionResult DiskDenseEngine::execute(const EngineExecutionRequest& request) const {
    const Layout layout = lay_out(shape_of(request));
    common::ensure(layout.reason.empty(),
                   "DiskDenseEngine::execute: estimate() must have been feasible for this request");
    const PassPlan& plan = layout.plan;
    const std::filesystem::path directory = request.scratch_directory.empty()
                                                ? std::filesystem::temp_directory_path()
                                                : request.scratch_directory;
    const FileBackingStore store(
        directory, checked_pow2(request.num_qubits), checked_pow2(plan.chunk_qubits));
    Window window(plan.window_qubits);
    initialize_store(store, window, request.global_phase);
    run_passes(store,
               window,
               plan,
               request.operations,
               request.monitor,
               request.max_threads,
               layout.fusion.has_value() ? &*layout.fusion : nullptr);

    EngineExecutionResult result;
    result.actual_bytes = window.size_bytes();
    if (request.want_counts) {
        result.counts = sample_store(store,
                                     window,
                                     request.shots,
                                     request.seed,
                                     request.measure_targets,
                                     request.bitstring_width);
    }
    if (request.want_expectation) {
        std::vector<std::complex<double>> terms;
        terms.reserve(request.observable->terms.size());
        for (const auto& term : request.observable->terms) {
            terms.push_back(
                expectation_of_term(store, window, term.first.x_words[0], term.first.z_words[0]));
        }
        result.expectation_terms = std::move(terms);
    }
    return result;
}

}  // namespace menos::engines::dense_tiered
