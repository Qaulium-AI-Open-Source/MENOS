#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "menos/circuit.h"
#include "menos/observable.h"

namespace menos {

// The circuit facts explain() reports, captured once at plan time from
// the original (pre-light-cone-reduction) circuit, so a plan-only
// preview and a post-execution report agree on what they mean by
// "circuit" (docs/BUILD_PLAN.md Phase 1, 1.4's explain() example:
// "qubits=5 clbits=5 operations=12 depth=6 global_phase=0"; Phase 7's
// Plan::explain()).
struct CircuitSummary {
    std::uint32_t num_qubits;
    std::uint32_t num_clbits;
    std::size_t num_operations;
    int depth;
    double global_phase;
    // Noise the job asked for (Phase 12, docs/adr/0016): its ChannelOps and
    // its measurements carrying a readout error. Both 0 for a noiseless job,
    // which is what keeps that job's explain() output unchanged.
    std::size_t num_noise_channels = 0;
    std::size_t num_readout_errors = 0;
};

// DirectDense: the dense engine runs the whole circuit as one region.
// DirectStabilizer: the stabilizer engine runs the whole circuit as one
// region (only feasible when every operation is Clifford and no
// statevector is requested). DirectPauli: the Pauli propagation engine
// runs the whole circuit as one region (only feasible for an expectation
// request within the term-count bound, docs/BUILD_PLAN.md Phase 5).
// DirectMPS: the MPS engine runs the whole circuit as one region (never
// feasible for a statevector request; under AccuracyPolicy::Exact, only
// when the rigorous Schmidt-rank bound fits budget; under Controlled,
// whenever a capped/truncated estimate fits, docs/BUILD_PLAN.md Phase
// 6). DirectTensorNetwork: the tensor-network engine closes the whole
// circuit's bra-ket-observable network to a scalar per term (only
// feasible for an expectation request whose contraction-order peak-tensor
// estimate fits budget, docs/BUILD_PLAN.md Phase 8; never counts or a
// statevector -- see the deviations table for why). Factorized: one
// region per interaction-graph component, executed sequentially, each
// region independently choosing its own cheapest eligible engine
// (docs/BUILD_PLAN.md Phase 3 and 4).
// CutAndRecombine: the circuit is split by wire cuts into fragments that are
// executed independently and recombined by a signed, weighted sum over the
// cut decomposition's branches (docs/adr/0013, Phase 9). Expectation requests
// only, and exact: every branch is enumerated, nothing is sampled.
// DirectDensityMatrix: the density-matrix engine runs the whole circuit as
// one region, exactly, noise included (Phase 12, docs/adr/0016). Enumerated
// only for a job that contains noise; last in the enum so every earlier
// value is unchanged.
// DirectTrajectory: the trajectory engine samples the noisy circuit as N
// independent pure-state trajectories (Phase 13, docs/adr/0017). Its results
// are statistical, never exact. Enumerated only for a noisy job whose caller
// opted in (RunOptions::expectation_precision or NoiseMethod::Auto); appended
// last so every earlier value is unchanged.
enum class Strategy {
    DirectDense,
    DirectStabilizer,
    DirectPauli,
    DirectMPS,
    DirectTensorNetwork,
    Factorized,
    CutAndRecombine,
    DirectDensityMatrix,
    DirectTrajectory
};
// DensityMatrix and Trajectory are the engines that can run a noise channel or
// a readout error: the first exactly (Phase 12, docs/adr/0016), the second by
// sampling (Phase 13, docs/adr/0017).
enum class Engine { Dense, Stabilizer, Pauli, MPS, TensorNetwork, DensityMatrix, Trajectory };
// The hardware a region executes on. Engine is the ALGORITHM (Dense,
// Stabilizer, ...); Backend is the machine it runs on. The planner chooses
// them separately: the algorithm by work units, then the backend by calibrated
// wall-clock seconds (docs/adr/0014).
enum class Backend { Cpu, Cuda };
// Where a region's state lives while it executes (Phase 23, docs/adr/0027).
// InMemory is every engine before Phase 23: the state is one buffer in the
// backend's own memory. Disk keeps the state in a scratch file and runs the
// gates chunk by chunk through a window in host memory; HostOffloaded keeps it
// in host memory and runs the chunks on the CUDA device (measured in step 1
// and not built, docs/adr/0027 section 10.1). A tier is a third coordinate of
// an implementation beside Engine and Backend: (Dense, Cpu, Disk) is the
// chunked executor over a scratch file, (Dense, Cpu, InMemory) is DenseEngine.
// The planner reaches for a tier other than InMemory only when the in-memory
// state does not fit (reach-only, section 9 Q5), so a job that fits is
// planned, executed and explained exactly as before.
enum class Tier { InMemory, Disk, HostOffloaded };

// Sentinel for Region::global_to_local: this global qubit does not belong
// to the region.
inline constexpr QubitId kNoLocalQubit = std::numeric_limits<QubitId>::max();

// What executing a candidate on an accelerator costs, beyond the work units and
// host bytes every candidate already reports (Phase 10, docs/adr/0014).
//
// These are separate fields rather than being folded into estimated_work_units
// because they are paid at different measured rates: bytes across the PCIe bus,
// work inside the GPU, and a fixed cost per kernel launch. Collapsing them into
// one number would make the GPU look uniformly fast, which is exactly the claim
// the Phase 10 gate warns against -- the launch term is what correctly makes a
// small circuit SLOWER on the GPU than on the CPU.
//
// Every field is zero for a CPU engine, so the planner's seconds formula
// degenerates to the pure work term for CPU candidates and is unchanged for
// them.
struct DeviceCost {
    // Device memory required. A budget entirely separate from host memory:
    // the development GPU offers 3681 MiB against a 7.75 GiB host budget, so a
    // state that fits comfortably in host memory can still be refused here.
    std::uint64_t device_bytes = 0;
    // Bytes crossing host-to-device. Small for this backend: the initial state
    // is built ON the device, so only gate matrices go up.
    std::uint64_t host_to_device_bytes = 0;
    // Bytes crossing device-to-host. 16 * 2^n when a statevector or counts are
    // wanted, and only a handful of bytes for an expectation-only job, which is
    // where the accelerator's advantage is largest.
    std::uint64_t device_to_host_bytes = 0;
    // Kernel launches. Each costs a fixed, measured overhead regardless of how
    // little work it does, which is what dominates a small circuit and is the
    // reason a CPU/GPU crossover exists at all.
    std::uint64_t kernel_launches = 0;
};

// What a candidate's runtime is made of, for the calibrated time model
// (Phase 24, docs/adr/0028 section 20). For each engine the candidate runs,
// up to kTimeComponents structural counts -- such as gates, gate updates or
// amplitude passes -- each one priced at its own rate, which
// plan::calibrated_hardware_profile measures on this machine by running that
// engine. Each engine's estimate() documents what its components count. A
// composite candidate (Factorized, CutAndRecombine) adds the terms of the
// regions and fragments it runs, each multiplied by how many times it runs.
//
// Kept apart from estimated_work_units, which stays the ranking unit for a
// job with no time budget, so that recalibrating time never changes such a
// job's plan (owner decision, docs/adr/0028 section 20). Doubles, because a
// component is a count multiplied by a run count and is only ever priced,
// never compared exactly. All zero for a candidate on a device or a tier
// other than InMemory, which are priced by their own measured rates.
inline constexpr std::size_t kTimeEngineCount = 7;  // the values of Engine
inline constexpr std::size_t kTimeComponents = 3;
struct TimeTerms {
    std::array<std::array<double, kTimeComponents>, kTimeEngineCount> counts{};
};

// A candidate's estimated resource requirements, or why it can't be
// costed (docs/BUILD_PLAN.md Phase 3, 3.5). estimated_peak_bytes and
// estimated_work_units are meaningful only when feasible is true;
// infeasible_reason is non-empty only when feasible is false. An
// unrepresentable estimate (checked-arithmetic overflow) is treated as
// infeasible with a documented reason, never a wraparound or an
// exception the caller must catch. Shared by the planner's top-level
// candidates and by SimulationEngine::estimate() (Phase 4 onward) -- the
// same shape serves both a whole-circuit candidate and a single region's
// per-engine estimate.
// What executing a candidate on a memory tier other than InMemory costs
// (Phase 23, docs/adr/0027 section 3): the state is read and written through
// the tier's link once per pass, which is what the planner prices with the
// link's measured bandwidth beside the work units. All zero for an in-memory
// engine, which never mentions it, so every pre-Phase-23 estimate is unchanged.
struct TierCost {
    // Gate passes over the whole state (docs/adr/0027 section 3.2), from the
    // circuit's own pass plan: what the executor will run, not a model of it.
    std::uint64_t passes = 0;
    // The chunk the executor chose, 2^chunk_qubits amplitudes, and its window
    // of 2^window_qubits amplitudes in host memory.
    std::uint32_t chunk_qubits = 0;
    std::uint32_t window_qubits = 0;
    // Bytes crossing the tier's link in all: the initial write of the state,
    // each pass's read and write, and the reads the outputs need.
    std::uint64_t link_bytes = 0;
    // The scratch file's size, 16 * 2^n, for the disk tier; 0 for a tier with
    // no file.
    std::uint64_t file_bytes = 0;
    // The part of link_bytes that is written: the initial write and each
    // pass's write-back. The rest of link_bytes is read. Kept apart because a
    // link's two directions are measured separately and can differ several
    // times over (Phase 24, docs/adr/0028 section 16.10.3).
    std::uint64_t link_write_bytes = 0;
};

// What gate fusion did to a dense region's sweep (Phase 24 item 7,
// docs/adr/0028 section 17): the blocks of two or more gates the containment
// rule formed, and the gates those blocks hold beyond their first, so the
// sweep makes `gates_absorbed` fewer passes than the gate count. Both zero for
// every non-dense engine, for a run with fusion off, and for a circuit in which
// nothing fused; explain() prints them only when gates_absorbed is not zero, so
// every other report is byte-identical to one from before fusion existed.
struct FusionSummary {
    std::uint64_t fused_blocks = 0;
    std::uint64_t gates_absorbed = 0;
};

struct CostEstimate {
    bool feasible;
    std::uint64_t estimated_peak_bytes;
    std::uint64_t estimated_work_units;
    std::string infeasible_reason;
    // Everything about running this candidate on an accelerator. All zero for
    // a CPU engine, which never mentions it (Phase 10, docs/adr/0014).
    DeviceCost device = {};
    // Everything about running this candidate on a memory tier. All zero for
    // an in-memory engine (Phase 23, docs/adr/0027).
    TierCost tier = {};
    // The fusion outcome this estimate charged (Phase 24 item 7). All zero
    // for every engine but the dense ones.
    FusionSummary fusion = {};
    // What the runtime is made of, for the calibrated time model (Phase 24,
    // docs/adr/0028 section 20). All zero when an engine does not report it,
    // in which case the planner prices estimated_work_units as before.
    TimeTerms time = {};
};

// A region is a contiguous local-index sub-circuit assigned an engine and
// backend. local_to_global[i] is the global qubit at local index i
// (ascending); global_to_local[g] is kNoLocalQubit unless global qubit g
// belongs to this region. operations are the circuit's own operations
// whose operands all lie in this region, remapped to local qubit indices
// (MeasureOp::clbit is left untouched -- clbits are a global,
// unpartitioned namespace, never remapped). DirectDense's one region has
// local_to_global == [0, 1, ..., num_qubits-1] (identity), and likewise
// for DirectStabilizer.
struct Region {
    std::vector<QubitId> local_to_global;
    std::vector<QubitId> global_to_local;
    std::vector<Operation> operations;
    Engine engine;
    Backend backend;
    // Where this region's state lives while it executes (Phase 23). InMemory
    // for every region of a job whose state fits its budget.
    Tier tier = Tier::InMemory;
    // The tier's own figures for this region (Phase 23, docs/adr/0027): the
    // pass plan's chunk and window the executor runs with, which the runtime
    // hands to the engine so the plan the planner priced is the plan that
    // runs. All zero for an in-memory region.
    TierCost tier_cost = {};
    // Why `engine` was chosen over the other available engine(s) for this
    // region, for explain() (docs/BUILD_PLAN.md Phase 4: mixed-engine
    // Factorized plans and the 99%-Clifford-plus-one-T case must show
    // their reasoning). Empty when there was no other eligible engine to
    // reject (e.g. a DirectDense/DirectStabilizer plan's one region).
    std::string engine_selection_reason;
    // This region's own restricted observable (common::restrict_observable,
    // docs/BUILD_PLAN.md Phase 5), set iff the job wants an expectation.
    // For a Direct* strategy's one region this is the whole job's
    // observable, unrestricted (nothing to restrict against).
    std::optional<Observable> observable;
    // 16 * 2^k bytes for this region's own state, or 0 for a region with
    // no operations and no measurement (no allocation needed -- its
    // outcome is statically |0>).
    std::uint64_t estimated_state_bytes;
    std::uint64_t estimated_work_units;
    // What gate fusion does to this region's sweep (Phase 24 item 7,
    // docs/adr/0028 section 17), from its engine's estimate: the engine
    // builds the same plan from the same operations when it runs. All zero
    // for a non-dense region, a run with fusion off, or a region in which
    // nothing fused.
    FusionSummary fusion = {};
    // Set only after execution (never at plan_job time -- there is
    // nothing to measure yet), and only for a region this engine == MPS
    // that actually truncated: the a-posteriori accumulated error bound
    // (EngineExecutionResult::accumulated_error_bound, docs/BUILD_PLAN.md
    // Phase 6). nullopt for every other engine, and for an MPS region
    // that never truncated. The runtime.cpp code path that executes this
    // region copies it here from the engine's own result before handing
    // the finished Plan to Result, mirroring how Plan::exact itself is
    // already only the pre-execution estimate while Result::exact()
    // holds the post-execution truth.
    std::optional<double> mps_accumulated_error_bound = std::nullopt;
    // Set only after execution, and only for a region whose engine ==
    // TensorNetwork: the actual peak intermediate tensor size (bytes,
    // complex128) reached during contraction, for comparison against the
    // pre-execution estimate already shown in the candidate table
    // (docs/BUILD_PLAN.md Phase 8's "estimated-versus-actual" explain()
    // requirement, mirroring mps_accumulated_error_bound's own identical
    // wiring pattern above). nullopt for every other engine.
    std::optional<std::uint64_t> tensor_network_actual_peak_bytes = std::nullopt;
    // This region's own predicted seconds on its backend, set by the planner
    // for each region of a Factorized plan when the profile is calibrated
    // (Phase 24, docs/adr/0028 section 16.4), so a replan can report each
    // region's prediction beside its observation. -1 otherwise.
    double estimated_seconds = -1.0;
    // The region's time terms, from its engine's estimate (Phase 24,
    // docs/adr/0028 section 20), which estimated_seconds is computed from
    // under a calibrated time model.
    TimeTerms time = {};
};

// What a CutAndRecombine plan did, for explain() (docs/adr/0013 Phase 9).
// Present only on a plan whose strategy is CutAndRecombine. branch_count is
// 8^K for K cuts, while distinct_evaluations is the far smaller number of
// fragment runs actually performed (4^m * 6^p summed over fragments) -- the
// gap between them is what makes cutting affordable, so explain() shows both
// rather than leaving the reader to assume they are equal.
// The engine one fragment evaluation runs on, and the peak it was costed at,
// chosen by plan_job (Phase 21, docs/adr/0025): the runtime runs each
// evaluation on exactly this engine and does no planning of its own.
struct CutEvaluation {
    Engine engine;
    std::uint64_t estimated_peak_bytes;
};

struct CutSummary {
    std::size_t cut_count;
    std::size_t fragment_count;
    std::uint64_t branch_count;
    std::uint64_t distinct_evaluations;
    // Measured after execution: the largest peak any single fragment actually
    // reached. nullopt on a plan-only preview.
    std::optional<std::uint64_t> actual_peak_bytes = std::nullopt;
    // evaluations[f][a]: fragment f's a-th distinct assignment, in
    // analysis::enumerate_assignments order.
    std::vector<std::vector<CutEvaluation>> evaluations = {};
};

// What a DirectTrajectory plan samples, for explain() and for the runtime
// (Phase 13, docs/adr/0017 section 3). Present only on a plan whose strategy is
// DirectTrajectory.
//
// trajectories is N = max(shots, N_eps), where N_eps = max(2, ceil(C^2 /
// eps^2)) for an expectation request with precision eps and C the sum of
// |coefficient| over the observable's non-identity terms. N_eps guarantees,
// before anything runs, that the true standard error is at most eps; the
// planner and the cost model use this same N. precision and coefficient_bound
// are set iff an expectation was requested.
struct TrajectorySummary {
    std::uint64_t trajectories;
    std::optional<double> precision;
    std::optional<double> coefficient_bound;
    // Measured after execution: the reported standard error of the expectation
    // value, s / sqrt(N). nullopt on a plan-only preview and for a counts-only
    // job.
    std::optional<double> standard_error = std::nullopt;
};

// One enumerated planner candidate's outcome, kept for every candidate
// (including the loser) so explain() can show why the winner won
// (docs/BUILD_PLAN.md Phase 3, 3.6).
struct CandidateSummary {
    Strategy strategy;
    bool feasible;
    std::string infeasible_reason;  // empty iff feasible
    std::uint64_t estimated_peak_bytes;
    std::uint64_t estimated_work_units;
    bool is_winner;
    // The backend this candidate was costed against. Cpu for every candidate
    // until the planner's backend-selection step runs, which may switch the
    // winner's to Cuda (docs/adr/0014). Recorded so explain() and the tests
    // can see which hardware the chosen plan is actually for.
    Backend backend = Backend::Cpu;
    // The memory tier this candidate was costed for (Phase 23). InMemory
    // unless the planner reached for a tier because the in-memory state does
    // not fit; explain() prints it only then, with the tier's figures below.
    Tier tier = Tier::InMemory;
    TierCost tier_cost = {};
    // The accelerator terms this candidate was costed with; all zero for a CPU
    // backend. Separate from estimated_peak_bytes, which is host memory. A
    // Factorized candidate stays on the Cpu backend even when some of its
    // regions run on CUDA; its device_bytes is then the largest such region's,
    // since regions run one at a time (docs/adr/0025 section 10.5).
    DeviceCost device = {};
    // estimated_work_units converted to wall-clock seconds using the
    // calibrated throughput of the machine that planned this job
    // (docs/BUILD_PLAN.md Phase 10). Negative means "not calibrated", which is
    // how a hand-built HardwareProfile in a lower-level test shows up; a job
    // carrying a time budget against such a profile is refused rather than
    // silently admitted. Meaningful only when feasible is true.
    double estimated_seconds = -1.0;
    // The estimate's time terms (Phase 24, docs/adr/0028 section 20), which
    // estimated_seconds is computed from when the profile carries a calibrated
    // time model.
    TimeTerms time = {};
};

// The light-cone reduction pass's own report (docs/BUILD_PLAN.md Phase
// 5): it runs once per job, unconditionally, before interaction-graph
// analysis. original == reduced counts when nothing was dropped (e.g. a
// plain statevector request, whose support is every qubit) -- still
// recorded, since the pass itself always runs.
struct LightConeSummary {
    std::uint32_t original_num_qubits;
    std::size_t original_num_operations;
    std::uint32_t reduced_num_qubits;
    std::size_t reduced_num_operations;
};

// The plan `Result::plan()` exposes: the winning strategy, exactness, its
// regions (component order; DirectDense has exactly one, covering every
// qubit), the winning candidate's totals, every enumerated candidate's
// summary, and the light-cone reduction report, for explain().
struct Plan {
    Strategy strategy;
    bool exact;
    std::vector<Region> regions;
    std::uint64_t estimated_state_bytes;
    std::uint64_t estimated_work_units;
    std::vector<CandidateSummary> candidates;
    LightConeSummary light_cone;
    // The original (pre-light-cone-reduction) circuit's own summary,
    // computed once by plan_job() (Phase 7) -- the single source both
    // Plan::explain() and Result::explain() read, so the two can never
    // report different "circuit" facts for the same plan.
    CircuitSummary circuit_summary;
    // The circuit every region above was actually built from (Phase 5:
    // light-cone-reduced, qubits possibly remapped -- see
    // docs/adr/0007). Every Region::local_to_global/global_to_local
    // pair is indexed against THIS circuit's qubit numbering, never the
    // original job's, so the runtime must read this (not the job's own
    // circuit) for anything qubit-numbering-sensitive: last-write-wins
    // resolution, check_plan_invariants, and the global statevector's
    // qubit count. The job's own global_phase is NOT here (light-cone
    // reduction always drops it, since it cancels in every requested
    // output) -- the runtime applies the original circuit's own
    // global_phase, unaffected by reduction. std::optional only so a
    // hand-built Plan in a lower-level unit test (one that never
    // reaches Runtime::execute()) is not forced to construct a Circuit
    // it has no use for; plan_job always populates it.
    std::optional<Circuit> reduced_circuit;
    // Set iff strategy == CutAndRecombine.
    std::optional<CutSummary> cut_summary = std::nullopt;
    // Set iff strategy == DirectTrajectory (Phase 13).
    std::optional<TrajectorySummary> trajectory_summary = std::nullopt;
    // The job's time budget, copied here so explain() can report both the
    // constraint and the estimates it was applied to (Phase 10).
    //
    // It also scopes explain()'s determinism. estimated_seconds below is a
    // MEASURED quantity and therefore differs between machines, so printing it
    // unconditionally would break the byte-identical report this type
    // promises. explain() prints the seconds columns only when this is set --
    // that is, only when the caller asked for a time-constrained plan, where a
    // machine-dependent answer is the whole point and is expected. Without a
    // budget the report is byte-identical to a pre-Phase-10 one.
    std::optional<double> time_budget_seconds = std::nullopt;
    // The backend the caller forced (RunOptions::backend, Phase 17), so
    // explain() can say the backend was requested rather than chosen. Unset
    // when the planner chose, and then explain() prints nothing extra.
    std::optional<Backend> requested_backend = std::nullopt;
    // The CUDA device's free memory this plan was checked against, read when
    // the job was planned (Phase 21, docs/adr/0025 section 9, Q3). Set only
    // when a candidate or a region of this plan runs on CUDA, so explain()
    // shows the figure a device decision rested on, and every plan with no
    // CUDA in it reports exactly as before. Like estimated_seconds, it is a
    // measured figure: it changes with other processes' use of the GPU.
    std::optional<std::uint64_t> device_available_bytes = std::nullopt;
    // The part of device_available_bytes that is MENOS's own memory pool,
    // retained from earlier jobs (Phase 22, docs/adr/0026 section 3.4), and
    // whether the pool is in use at all (false on a device without
    // memory-pool support). Set exactly when device_available_bytes is.
    std::optional<std::uint64_t> device_pool_bytes = std::nullopt;
    std::optional<bool> device_pooled = std::nullopt;
    // The memory tier the caller forced (RunOptions::tier, Phase 23,
    // docs/adr/0027), so explain() can say the tier was requested rather than
    // reached. Unset when the planner chose, and then explain() prints
    // nothing extra.
    std::optional<Tier> requested_tier = std::nullopt;
    // The scratch directory a disk-tier decision rested on, with the space
    // available on it and its measured bandwidth (0 when unmeasured), read
    // when the job was planned. Set only when a candidate or a region of this
    // plan is on the disk tier, so every plan with no disk tier in it reports
    // exactly as before. Like device_available_bytes, these are measured
    // figures of the moment.
    std::optional<std::filesystem::path> scratch_directory = std::nullopt;
    std::optional<std::uint64_t> disk_available_bytes = std::nullopt;
    std::optional<double> disk_write_bytes_per_second = std::nullopt;
    std::optional<double> disk_read_bytes_per_second = std::nullopt;

    // A deterministic text report: the "circuit", "light_cone",
    // candidate, "strategy" and region lines, and the "estimated" line
    // -- no "actual" section, since nothing has executed
    // (docs/BUILD_PLAN.md Phase 7). Byte-identical to the same lines in
    // Result::explain() for a Result built from executing this same
    // Plan; Result::explain() calls this directly and appends its own
    // "actual"/"expectation" lines, so the two cannot drift apart.
    [[nodiscard]] std::string explain() const;
};

}  // namespace menos
