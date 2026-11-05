#pragma once

#include <algorithm>
#include <compare>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

#include "menos/plan.h"

namespace menos::plan {

// One region of a plan, identified within its job (Phase 24, docs/adr/0028
// sections 16.4 and 16.8): the region's qubits (the planned circuit's, in
// ascending order) and the implementation that runs it. Two regions of the same
// size but different qubits are different keys, so a region's floor is never
// applied to any region but itself.
struct RegionKey {
    std::vector<QubitId> qubits;
    Engine engine;
    Backend backend;
    Tier tier;
    auto operator<=>(const RegionKey&) const = default;
};

[[nodiscard]] inline RegionKey region_key(const Region& region) {
    RegionKey key{region.local_to_global, region.engine, region.backend, region.tier};
    std::sort(key.qubits.begin(), key.qubits.end());
    return key;
}

// A whole plan, identified within its job by its strategy and every region's
// key, idle regions included: the plan floor of section 16.8.
struct PlanKey {
    Strategy strategy;
    std::vector<RegionKey> regions;
    auto operator<=>(const PlanKey&) const = default;
};

[[nodiscard]] inline PlanKey plan_key(Strategy strategy, const std::vector<Region>& regions) {
    PlanKey key{strategy, {}};
    key.regions.reserve(regions.size());
    for (const Region& region : regions) {
        key.regions.push_back(region_key(region));
    }
    return key;
}

[[nodiscard]] inline PlanKey plan_key(const Plan& plan) {
    return plan_key(plan.strategy, plan.regions);
}

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
// Scope, stated rather than left to be discovered: a floor in
// observed_floor_seconds is attributed to one (engine, backend, tier) triple,
// so it applies to the single-engine Direct* candidates only. From Phase 24 a
// Factorized plan's overrun is attributed per region (region_floor_seconds)
// and to the plan as a whole (plan_floor_seconds), below. A CutAndRecombine
// plan's overrun is still not attributed, and is refused. The tier (Phase 23,
// docs/adr/0027) is part of the key because an overrun of the disk tier says
// nothing about the in-memory engine of the same (engine, backend): they are
// different implementations with different costs.
struct RuntimeObservations {
    std::map<std::tuple<Engine, Backend, Tier>, double> observed_floor_seconds;
    // Phase 24 (docs/adr/0028 sections 16.4 and 16.8), for a Factorized plan
    // that overran. A region floor is the time one region of it took on its
    // implementation (the whole region if it finished, up to the crossing if
    // it was running): a lower bound for that region, on that implementation,
    // and for nothing else. A plan floor is the time the whole plan took
    // before it was abandoned: with it, that exact plan's prediction exceeds
    // the budget for the rest of the job, which is what makes the replan loop
    // terminate. Both are raise-only, like the floors above.
    std::map<RegionKey, double> region_floor_seconds;
    std::map<PlanKey, double> plan_floor_seconds;

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

    [[nodiscard]] std::optional<double> region_floor_for(const RegionKey& key) const {
        const auto it = region_floor_seconds.find(key);
        if (it == region_floor_seconds.end()) {
            return std::nullopt;
        }
        return it->second;
    }
    void observe_region(const RegionKey& key, double seconds) {
        double& floor = region_floor_seconds[key];
        floor = std::max(floor, seconds);
    }

    [[nodiscard]] std::optional<double> plan_floor_for(const PlanKey& key) const {
        const auto it = plan_floor_seconds.find(key);
        if (it == plan_floor_seconds.end()) {
            return std::nullopt;
        }
        return it->second;
    }
    void observe_plan(const PlanKey& key, double seconds) {
        double& floor = plan_floor_seconds[key];
        floor = std::max(floor, seconds);
    }

    [[nodiscard]] bool empty() const {
        return observed_floor_seconds.empty() && region_floor_seconds.empty() &&
               plan_floor_seconds.empty();
    }
};

}  // namespace menos::plan
