// What two planners are compared on (Phase 24, docs/adr/0028 sections 3 and
// 6): Plan::explain() plus the plan fields explain() does not print, or the
// refusal a job ends in.
//
// Shared by the golden-file test (tests/plan/planner_golden_test.cpp) and the
// differential test against the frozen d89dde3 planner
// (tests/plan/planner_differential_test.cpp). The frozen planner compiles this
// same header against its own d89dde3 sources (tests/plan/frozen_d89dde3), so
// both sides print a plan with the same code.

#pragma once

#include <format>
#include <string>

#include "common/error.h"
#include "menos/plan.h"
#include "plan/planner.h"
#include "planner_corpus.h"

namespace menos::planner_corpus {

// explain() plus what it leaves out: each region's operation count, observable
// term count and estimates, and the reduced circuit's size.
inline std::string plan_fingerprint(const Plan& plan) {
    std::string out = plan.explain();
    out += "--- plan fields not printed by explain() ---\n";
    for (std::size_t i = 0; i < plan.regions.size(); ++i) {
        const Region& region = plan.regions[i];
        out += std::format(
            "region {}: operations={} observable_terms={} state_bytes={} "
            "work_units={} global_to_local_size={}\n",
            i,
            region.operations.size(),
            region.observable.has_value() ? region.observable->terms.size() : 0,
            region.estimated_state_bytes,
            region.estimated_work_units,
            region.global_to_local.size());
    }
    out += std::format("reduced_circuit: {}\n",
                       plan.reduced_circuit.has_value()
                           ? std::format("qubits={} operations={}",
                                         plan.reduced_circuit->num_qubits(),
                                         plan.reduced_circuit->operations().size())
                           : std::string("none"));
    return out;
}

// The plan's fingerprint, or the exception plan_job refused the job with. The
// exception type is spelled out by hand, never taken from typeid: the frozen
// planner's types live in a renamed namespace.
inline std::string outcome_of(const CorpusJob& job) {
    try {
        return plan_fingerprint(plan::plan_job(job.spec, job.hardware));
    } catch (const common::ResourceExceeded& e) {
        return std::string("THROWS ResourceExceeded: ") + e.what() + "\n";
    } catch (const common::UnsupportedFeature& e) {
        return std::string("THROWS UnsupportedFeature: ") + e.what() + "\n";
    } catch (const common::InvalidCircuit& e) {
        return std::string("THROWS InvalidCircuit: ") + e.what() + "\n";
    } catch (const common::InternalError& e) {
        return std::string("THROWS InternalError: ") + e.what() + "\n";
    }
}

}  // namespace menos::planner_corpus
