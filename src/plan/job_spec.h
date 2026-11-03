#pragma once

#include <cstdint>
#include <optional>

#include "common/thread_count.h"
#include "menos/accuracy_policy.h"
#include "menos/circuit.h"
#include "menos/noise_method.h"
#include "menos/observable.h"
#include "menos/plan.h"

namespace menos::plan {

// circuit, output request (statevector yes/no, shots, seed, expectation),
// resource budgets (memory bytes and wall-clock seconds, both optional),
// accuracy policy (Phase 6).
struct JobSpec {
    Circuit circuit;
    std::uint64_t shots;
    bool want_statevector;
    std::uint64_t seed;
    std::optional<std::uint64_t> memory_budget_bytes;
    std::optional<Observable> expectation;
    AccuracyPolicy accuracy_policy = {};
    // Wall-clock seconds this job may take, enforced at PLAN time against the
    // calibrated throughput (docs/BUILD_PLAN.md Phase 10, docs/adr/0014).
    //
    // Semantics, deliberately strict: a candidate whose estimated runtime
    // exceeds this is INFEASIBLE, exactly as an over-budget memory estimate
    // is, and if no candidate fits, planning throws rather than running the
    // cheapest one anyway. It is a constraint, never a best-effort hint, and
    // it is never silently downgraded to one.
    //
    // From Phase 11 (docs/adr/0015) it bounds the EXECUTION too, not only the
    // estimate. A run whose estimate fit but whose observed elapsed time
    // exceeds this is abandoned at a safe boundary; the planner is then re-run
    // with the observed cost of the engine that overran, and either a different
    // plan takes over or the job is refused with ResourceExceeded. Nothing is
    // silently allowed to overrun any more.
    //
    // Precisely what it bounds, because this is the one place it could be
    // overread: ONE execution attempt. A job that replanned ran more than one,
    // so its total wall time is larger than this figure -- the abandoned
    // attempt's seconds are reported in Result::replans() rather than hidden.
    // Bounding the sum instead was tried and is self-defeating: a crossing
    // means the budget is already spent, so no replan could ever complete.
    //
    // Planning itself is not charged against it (bounded, and measured between
    // 0.05 ms and 85 ms).
    std::optional<double> time_budget_seconds = std::nullopt;
    // Phase 13 opt-ins for trajectory sampling (RunOptions has the full
    // semantics, docs/adr/0017 section 9). Defaults keep Phase 12 planning.
    NoiseMethod noise_method = NoiseMethod::Exact;
    std::optional<double> expectation_precision = std::nullopt;
    // CPU threads for gate application, resolved (Phase 15, docs/adr/0019
    // section 3.3): in [1, hardware threads]. Defaults to every hardware
    // thread, so a JobSpec built anywhere other than menos::run (batch,
    // mitigation, tests) gets RunOptions' default rather than one thread.
    std::uint32_t max_threads = common::default_thread_count();
    // RunOptions::fuse_diagonal_gates, read by the CPU dense engine only.
    bool fuse_diagonal_gates = false;
    // A forced backend (RunOptions::backend has the semantics, docs/adr/0021
    // section 2). Unset: the planner chooses, as before.
    std::optional<Backend> backend = std::nullopt;
};

}  // namespace menos::plan
