#pragma once

#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "common/rng.h"
#include "common/thread_count.h"
#include "menos/accuracy_policy.h"
#include "menos/bitstring.h"
#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/plan.h"

namespace menos {

// What a SimulationEngine can produce. A region's requested output must
// be a subset of its assigned engine's capabilities -- the planner
// enforces this via estimate() returning infeasible for a request an
// engine cannot satisfy (docs/BUILD_PLAN.md Phase 4/5).
struct EngineCapabilities {
    bool counts;
    bool statevector;
    bool expectation;
};

// A region's outcome distribution, exposed only for the multi-region
// Factorized counts-combination path (src/runtime/combine_counts.h): the
// single-region case (is_sole_region == true) gets a complete histogram
// directly instead (EngineExecutionResult::counts), avoiding a per-shot
// virtual call there. rng is the region's own common::Rng stream, shared
// across every shot; implementations draw as many words as they need
// from it per call (dense: one word, converted via
// common::Rng::word_to_uniform for the existing cumulative-distribution
// walk; stabilizer: ceil(r/64) words, whose raw bits select a subset of
// the affine solution space's up-to-n free generators -- a single word
// is not always enough, unlike dense's one-uniform-per-shot need).
class RegionCountsSampler {
public:
    virtual ~RegionCountsSampler() = default;

    [[nodiscard]] virtual BitString sample_one(common::Rng& rng) const = 0;
};

// Observes an execution in progress and may abandon it (docs/adr/0015,
// Phase 11).
//
// An engine calls at_safe_boundary() at each of ITS OWN safe boundaries. What
// counts as one differs by engine and by phase of the work:
//   * between two operations, for every engine that sweeps operations;
//   * between two observable terms, for the tensor-network engine, whose
//     execute() is a contraction rather than a sweep;
//   * between two sampled shots, where drawing one shot is itself expensive
//     (the MPS sampler walks the chain per shot, and measurement showed
//     sampling to be 69% of an MPS run's time -- leaving it unmonitored would
//     have let a job overrun its budget in a phase nothing was watching).
// The dense sampler is deliberately NOT monitored per shot: it is one O(2^n)
// cumulative pass, the same order as a single gate, so it cannot overrun by
// more than one gate's granularity, and a per-basis-state check would cost
// 16 million clock reads at 24 qubits.
//
// A boundary is safe because the engine's
// state is complete and self-consistent there, nothing has been handed to the
// caller, and every allocation the engine owns is released by its own
// destructor if the stack unwinds.
//
// To abandon, an implementation THROWS. There is no stop-and-return protocol:
// a return value would have to be threaded through every engine's inner loop
// and checked by hand at each level, and forgetting one would silently ignore
// the abandon request. Throwing cannot be silently ignored. Implementations
// that do not abandon simply return.
//
// Engines never interpret the monitor, construct one, or know what a threshold
// is; they only report reaching a boundary. The decision lives entirely in
// src/api (menos::api::BudgetMonitor), which is the layer that owns both the
// planner and the runtime.
//
// boundaries_crossed counts boundaries reached within the CURRENT engine
// invocation, starting at 1, so a Factorized plan restarts the count at each
// region. It is reported for explain(); it is deliberately NOT turned into a
// progress fraction or used to project a finish time, both because the boundary
// kinds above are not commensurable and because measurement showed a rate taken
// from partial progress mispredicts the total by 0.16x-0.40x on the MPS engine
// (docs/adr/0015).
class ExecutionMonitor {
public:
    ExecutionMonitor() = default;
    virtual ~ExecutionMonitor() = default;
    ExecutionMonitor(const ExecutionMonitor&) = delete;
    ExecutionMonitor& operator=(const ExecutionMonitor&) = delete;
    ExecutionMonitor(ExecutionMonitor&&) = delete;
    ExecutionMonitor& operator=(ExecutionMonitor&&) = delete;

    // Returns normally to continue; throws to abandon the run.
    virtual void at_safe_boundary(std::uint64_t boundaries_crossed) const = 0;
};

// One region's execution request. operations are this region's own
// operations, already remapped to local qubit indices (MeasureOp::clbit
// is left untouched -- see menos::Region's own doc comment).
// measure_targets is the already-resolved (global last-write-wins
// applied) (local qubit, global clbit) mapping for this region.
// bitstring_width is circuit.num_clbits() (a global width, shared by
// every region). global_phase is the phase this engine should apply
// while initializing its own state: circuit.global_phase for a sole
// region (DirectDense/DirectStabilizer, where this engine's output is
// the final result), or 0.0 for a Factorized region (the runtime applies
// the phase once, after recombination, over the global buffer -- see
// docs/adr/0005's global-phase bug fix; an engine with no statevector
// capability, i.e. the stabilizer engine, never observes a nonzero value
// here since global phase cannot affect any measurement probability it
// produces). budget_bytes is this region's own estimated_state_bytes
// from planning, re-checked (not re-decided) at execution time.
// observable is this region's own restricted observable (docs/adr/0007:
// each term's PauliString already has its bits projected down to just
// this region's local qubits by the caller -- an engine never sees a
// term wider than its own region). Non-owning, non-null iff
// want_expectation. accuracy_policy defaults to Exact, so every
// pre-Phase-6 construction site keeps its current (exact) behavior
// unchanged; only DirectMPS reads it (docs/BUILD_PLAN.md Phase 6).
struct EngineExecutionRequest {
    std::uint32_t num_qubits;
    std::span<const Operation> operations;
    bool want_statevector;
    bool want_counts;
    bool want_expectation;
    std::uint64_t shots;
    std::span<const std::pair<QubitId, ClbitId>> measure_targets;
    std::uint32_t bitstring_width;
    std::uint64_t seed;
    double global_phase;
    std::uint64_t budget_bytes;
    bool is_sole_region;
    const Observable* observable;
    AccuracyPolicy accuracy_policy = {};
    // Non-owning, null unless the job carries a time budget (Phase 11,
    // docs/adr/0015). Null is the overwhelmingly common case and costs an
    // engine one predictable, well-predicted branch per boundary; a monitored
    // run additionally pays one virtual call and one clock read per boundary:
    // 31.4 ns against a cheapest case of ~1.3 us of work per boundary, so at
    // most ~2.4% on the cheapest engine, and too small to resolve above
    // run-to-run noise even there.
    const ExecutionMonitor* monitor = nullptr;
    // The trajectory count N the plan fixed (Plan::trajectory_summary, Phase
    // 13, docs/adr/0017 section 3). Read only by the trajectory engine; 0 for
    // every other engine.
    std::uint64_t trajectories = 0;
    // The most CPU threads the engine may use for gate application (Phase 15,
    // docs/adr/0019 section 3), already resolved from RunOptions::max_threads
    // to a value in [1, hardware threads]. Read by the CPU statevector,
    // density-matrix and trajectory engines; results do not depend on it.
    // Defaults to every hardware thread, RunOptions' own default, so a request
    // built directly (calibration, benchmarks, tests) measures and exercises
    // the configuration jobs run in rather than silently running on one.
    std::uint32_t max_threads = common::default_thread_count();
    // RunOptions::fuse_diagonal_gates (src/ir/fuse_diagonal.h). Read by the
    // CPU dense engine only; every other engine applies the operations as
    // given. Off unless the job asked for it.
    bool fuse_diagonal_gates = false;
};

// counts is populated iff request.want_counts && request.is_sole_region;
// counts_sampler is populated iff request.want_counts &&
// !request.is_sole_region; statevector is populated iff
// request.want_statevector (only the dense engine ever sets it).
// expectation_terms is populated iff request.want_expectation: one raw
// <psi|P|psi> per term of request.observable, in the same order,
// coefficient NOT applied (docs/adr/0007) -- coefficients are combined
// by the caller (src/runtime/combine_expectation.h), which for a
// Factorized plan must take the PRODUCT of each region's own raw value
// for a given term before multiplying by that term's coefficient
// (mixed-product property), not a per-region weighted sum. A
// DirectDense/DirectStabilizer/DirectPauli region is a degenerate
// single-region product, combined the same way. actual_bytes is a
// genuine measurement of this region's own peak allocation, not a reuse
// of the estimate. accumulated_error_bound is populated only by an MPS
// region that actually truncated: the a-posteriori bound built from
// this run's own per-truncation-step discarded weights (CLAUDE.md's
// "Controlled(tolerance) is a per-step cutoff, not a global budget"
// domain invariant) -- nullopt for every other engine and for an MPS
// region that never truncated.
//
// statistical is true only for an engine whose outputs are sampled estimates
// rather than exact for the requested model: the trajectory engine (Phase 13,
// docs/adr/0017). Its expectation_terms are then per-term SAMPLE MEANS over
// trajectories (still coefficient-free, so the caller combines them exactly as
// above -- the mean is linear), and expectation_standard_error is the standard
// error of the coefficient-weighted expectation, s / sqrt(N), measured from the
// per-trajectory values. It cannot be rebuilt from the per-term means, which is
// why the engine reports it. Both stay at their defaults for every exact engine.
struct EngineExecutionResult {
    std::optional<std::vector<std::complex<double>>> statevector;
    std::optional<std::map<BitString, std::uint64_t>> counts;
    std::unique_ptr<RegionCountsSampler> counts_sampler;
    std::optional<std::vector<std::complex<double>>> expectation_terms;
    std::uint64_t actual_bytes;
    std::optional<double> accumulated_error_bound;
    bool statistical = false;
    std::optional<double> expectation_standard_error = std::nullopt;
};

// The output request an engine's estimate() costs: which qubit count
// and operations, and which outputs are wanted. operations is
// non-owning (matches the shape estimate()'s callers already build
// these from, e.g. cost::pick_cheapest_engine's per-component loop).
// observable is non-owning, non-null iff want_expectation. Like
// operations, its caller decides the scope: the whole job's observable
// for a DirectDense/DirectStabilizer/DirectPauli candidate, or a
// region-restricted one (mirroring how operations is already
// region-restricted there) inside estimate_factorized's per-component
// loop. accuracy_policy defaults to Exact -- see EngineExecutionRequest's
// own doc comment for why this keeps every pre-Phase-6 site unchanged.
// budget_bytes defaults to unbounded and, like accuracy_policy, is read
// by only one engine. Every engine before TensorNetworkEngine computes its
// own cost from a closed-form or linear-time formula, so its estimate() is
// cheap regardless of the answer, and the generic budget-versus-estimate
// comparison already happens uniformly after estimate() returns, in
// cost::estimate_direct/estimate_factorized. TensorNetworkEngine::estimate()
// is different: it runs a real elimination search whose own cost is a
// function of the graph's structure, not bounded by the job's budget the way
// every other engine's cost already is -- so it reads this field to bound
// the search itself, stopping as soon as the running width already implies
// more than budget_bytes. This is the principled alternative to special-casing
// the candidate in the planner: the search is bounded, the candidate still
// competes on cost like every other one (docs/BUILD_PLAN.md Phase 8).
struct EngineEstimateRequest {
    std::uint32_t num_qubits;
    const std::vector<Operation>* operations;
    bool want_statevector;
    bool want_counts;
    bool want_expectation;
    const Observable* observable;
    AccuracyPolicy accuracy_policy = {};
    std::uint64_t budget_bytes = std::numeric_limits<std::uint64_t>::max();
    // The trajectory count N to cost (Phase 13, docs/adr/0017 section 5). Read
    // only by the trajectory engine, whose cost is N times one trajectory's;
    // 0 for every other engine, and the trajectory engine reports 0 as
    // infeasible rather than costing an empty run.
    std::uint64_t trajectories = 0;
    // Branch-and-bound bound (planning speed only, never a result change).
    // When set below its default, the caller already holds a feasible
    // alternative costing exactly this many work units, so an engine whose
    // estimate() builds its figure up incrementally MAY stop as soon as its
    // running total is strictly greater than this, returning
    // CostEstimate::feasible = false with a reason that starts with
    // kPrunedReasonPrefix. Strictly greater: running totals only grow, so such
    // a candidate could never have been strictly cheaper than the alternative,
    // and ties already go to the earlier-enumerated candidate. An engine may
    // ignore the field; the default never prunes.
    std::uint64_t prune_above_work_units = std::numeric_limits<std::uint64_t>::max();
};

// Prefix of every infeasible_reason produced by the bound above.
inline constexpr std::string_view kPrunedReasonPrefix = "pruned by branch-and-bound";

// A simulation engine: something that can estimate the cost of, and
// execute, a region (docs/BUILD_PLAN.md Phase 4). The planner sees
// engines only through this interface, never through DenseEngine or
// StabilizerEngine directly (CLAUDE.md's architecture rule) -- the only
// place that names both concrete engines is src/plan/available_engines.h.
class SimulationEngine {
public:
    virtual ~SimulationEngine() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;
    // The menos::Engine enum value identifying this engine, so
    // Region::engine can be set from a SimulationEngine* without any
    // caller needing to know available_engines()'s positional order.
    [[nodiscard]] virtual Engine engine_id() const = 0;
    // The hardware this implementation runs on. Pure virtual, like
    // engine_id(), so every engine states it rather than inheriting a default
    // that would silently be wrong for a new backend. Two implementations of
    // the same Engine may differ only in this (Phase 10: DenseEngine is
    // {Dense, Cpu}, CudaDenseEngine is {Dense, Cuda}), which is what lets the
    // runtime resolve a Region's (engine, backend) pair to exactly one
    // implementation.
    [[nodiscard]] virtual Backend backend() const = 0;
    [[nodiscard]] virtual EngineCapabilities capabilities() const = 0;

    // The work-unit/peak-byte cost of running request.operations on an
    // m-qubit region and producing the requested outputs, or why this
    // engine cannot do so (e.g. a non-Clifford operation for the
    // stabilizer engine, or want_statevector for an engine whose
    // capabilities().statevector is false). Never allocates region
    // state, and never throws: an unrepresentable estimate (checked-
    // arithmetic overflow, or a Pauli-propagation term count that
    // overflows before propagation ever starts) is returned as an
    // infeasible CostEstimate with a documented reason, exactly like any
    // other infeasibility.
    [[nodiscard]] virtual CostEstimate estimate(const EngineEstimateRequest& request) const = 0;

    // Runs request.operations and produces the requested outputs.
    // Precondition: estimate() for the equivalent EngineEstimateRequest
    // returned feasible -- not re-checked here beyond what allocation
    // itself enforces.
    [[nodiscard]] virtual EngineExecutionResult execute(
        const EngineExecutionRequest& request) const = 0;
};

}  // namespace menos
