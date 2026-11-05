#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>

#include "menos/plan.h"

namespace menos::plan {

// What a run that was abandoned mid-flight taught the planner (Phase 11,
// docs/adr/0015).
//
// This is NOT a second cost model and NOT a recalibration. It carries one kind
// of fact, and that fact is a LOWER BOUND that was directly observed: engine E
// on backend B, on this exact job, has already spent this many seconds. The
// planner takes max(model estimate, observed floor), so an observation can only
// ever make a candidate look more expensive, never cheaper. A wrong observation
// cannot make the planner optimistic.
//
// Why a floor rather than a measured rate. The obvious alternative -- divide
// observed seconds by completed work units to get a per-engine rate, then
// re-estimate -- was measured and rejected. The MPS engine's seconds per work
// unit swings 30x WITHIN a single run as its bond dimension grows, so a rate
// taken from partial progress projects the total at 0.16x to 0.40x of the truth
// (docs/adr/0015). A projection built on it would under-predict exactly where
// the planner is already wrong. Elapsed-so-far needs no such assumption: it is
// true by construction, and it is monotone, which is what makes the replan loop
// terminate.
//
// Scope, stated rather than left to be discovered: a floor is attributed to one
// (engine, backend, tier) triple, so it applies to the single-engine Direct*
// candidates only. A Factorized or CutAndRecombine candidate spans several
// engines and this phase does not attribute elapsed time per region, so no
// floor is applied to it. Both of those models were measured accurate to
// within 0.70x-1.33x, so they are not where the problem is. The tier (Phase 23,
// docs/adr/0027) is part of the key because an overrun of the disk tier says
// nothing about the in-memory engine of the same (engine, backend): they are
// different implementations with different costs.
struct RuntimeObservations {
    std::map<std::tuple<Engine, Backend, Tier>, double> observed_floor_seconds;

    // The floor for this triple, or nullopt if it was never observed.
    [[nodiscard]] std::optional<double> floor_for(Engine engine,
                                                  Backend backend,
                                                  Tier tier = Tier::InMemory) const {
        const auto it = observed_floor_seconds.find({engine, backend, tier});
        if (it == observed_floor_seconds.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    // Records that (engine, backend, tier) has already spent `seconds` on this
    // job. Keeps the LARGER of any existing floor and this one, so accumulating
    // observations across successive abandoned attempts can never walk a floor
    // backwards.
    void observe(Engine engine, Backend backend, Tier tier, double seconds) {
        double& floor = observed_floor_seconds[{engine, backend, tier}];
        floor = std::max(floor, seconds);
    }
    // The in-memory implementation's floor: what every caller before Phase 23
    // observed.
    void observe(Engine engine, Backend backend, double seconds) {
        observe(engine, backend, Tier::InMemory, seconds);
    }

    [[nodiscard]] bool empty() const {
        return observed_floor_seconds.empty();
    }
};

}  // namespace menos::plan
