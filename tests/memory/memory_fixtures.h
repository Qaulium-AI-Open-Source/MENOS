#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "menos/accuracy_policy.h"
#include "menos/circuit.h"
#include "menos/observable.h"
#include "menos/plan.h"

// The Phase 21 memory fixture set (docs/adr/0025 section 4): every engine and
// output kind, run either through one SimulationEngine directly (so the
// engine's own estimate is compared with what its execute() allocates) or
// through menos::run (so the plan's estimate is compared with the whole job).
// Shared by menos_memory_tests and menos_memory_probe, which must run exactly
// the same thing for the heap-tracker and RSS figures to be comparable.
namespace menos::test::memory {

enum class FixtureKind { Engine, Run, Mitigated };

struct Fixture {
    std::string name;
    FixtureKind kind;
    // Engine fixtures: "dense", "dense_cuda", "density_matrix", "trajectory",
    // "stabilizer", "pauli", "mps", "tensor_network". Also used to group the
    // RSS baseline B (a 1-qubit run of the same engine).
    std::string engine;
    Circuit circuit;
    bool want_statevector = false;
    std::uint64_t shots = 0;
    std::optional<Observable> observable = std::nullopt;
    AccuracyPolicy accuracy_policy = {};
    std::uint64_t trajectories = 0;
    // false: the engine returns a counts sampler (a Factorized region), which
    // is drawn sampler_draws times while the result is still held, as the
    // runtime does.
    bool sole_region = true;
    std::uint64_t sampler_draws = 0;
    // Run fixtures: the strategy the planner is expected to choose, so a
    // fixture cannot silently stop exercising the path it is named for.
    std::optional<Strategy> expected_strategy = std::nullopt;
    // A 1-qubit run used only as the RSS baseline B of its engine.
    bool rss_baseline = false;
    // Too large for the test binary (the n = 24 gate sizes); probe only.
    bool probe_only = false;
};

// Every fixture this build can run: the CUDA ones only when built with CUDA.
[[nodiscard]] const std::vector<Fixture>& fixtures();

[[nodiscard]] const Fixture* find_fixture(std::string_view name);

// The RSS baseline fixture of fixture's engine (a Run fixture's baseline is the
// 1-qubit menos::run).
[[nodiscard]] const Fixture& rss_baseline_for(const Fixture& fixture);

struct Measurement {
    // The engine's estimated_peak_bytes, or the winning candidate's for a run.
    std::uint64_t estimated_peak_bytes = 0;
    std::uint64_t estimated_device_bytes = 0;
    // EngineExecutionResult::actual_bytes, or Result::actual_state_bytes().
    std::uint64_t reported_actual_bytes = 0;
    // Heap-tracker peak above the live bytes at entry; 0 if no tracker.
    std::uint64_t tracked_peak_bytes = 0;
    std::string strategy;
    // FNV-1a 64-bit digests of the outputs' exact bits, as 16 hex digits, or
    // "-" for an output the fixture does not request.
    std::string counts_digest = "-";
    std::string statevector_digest = "-";
    std::string expectation_digest = "-";
};

// False when the fixture needs a device this process does not have.
[[nodiscard]] bool can_run(const Fixture& fixture);

// Does the one-time work a fixture's run would otherwise include: for a job,
// the planner's per-process hardware calibration, whose buffers would
// otherwise dominate a small job's peak. run_fixture() calls it too.
void prepare(const Fixture& fixture);

// Runs fixture once and measures it with the heap tracker linked into the
// calling executable.
[[nodiscard]] Measurement run_fixture(const Fixture& fixture);

}  // namespace menos::test::memory
