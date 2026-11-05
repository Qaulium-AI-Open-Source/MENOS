#pragma once

#include "menos/plan.h"
#include "menos/result.h"
#include "menos/simulation_engine.h"
#include "plan/job_spec.h"

namespace menos::runtime {

class Runtime {
public:
    // `monitor` is handed to every engine this plan invokes, unchanged (Phase
    // 11, docs/adr/0015). Null -- the default, and every caller that sets no
    // time budget -- means no engine observes anything and execution is
    // identical to before.
    //
    // The runtime neither creates a monitor nor interprets one. It does not
    // decide what a threshold is, does not replan, and does not catch the
    // monitor's abandon signal: that signal passes straight through to
    // src/api, which is the layer that owns both the planner and the runtime.
    // Keeping the decision there is what stops a second planner appearing in
    // here.
    //
    // The Result takes the plan over (Phase 21, docs/adr/0025): on return,
    // `plan` has been moved into the Result, with this execution's own
    // figures recorded on it (MPS error bounds, tensor-network and cut peaks,
    // the trajectory standard error), so the job holds one copy of its plan,
    // not two. If execution throws -- an abandoned attempt included -- `plan`
    // is left exactly as it was, which is what lets the replan loop describe
    // the plan it is replacing.
    [[nodiscard]] Result execute(Plan&& plan,
                                 const menos::plan::JobSpec& job_spec,
                                 const ExecutionMonitor* monitor = nullptr) const;

    // The same, for a caller that keeps its plan: runs a copy of it.
    [[nodiscard]] Result execute(const Plan& plan,
                                 const menos::plan::JobSpec& job_spec,
                                 const ExecutionMonitor* monitor = nullptr) const;
};

}  // namespace menos::runtime
