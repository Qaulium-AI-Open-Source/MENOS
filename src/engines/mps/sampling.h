#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "common/rng.h"
#include "engines/mps/mps_state.h"
#include "menos/bitstring.h"
#include "menos/circuit.h"
#include "menos/simulation_engine.h"

namespace menos::engines::mps {

// Sequential conditional sampling down the chain (docs/BUILD_PLAN.md
// Phase 6): takes the state (the engine moves its own in, docs/adr/0025)
// and moves its orthogonality center to site 0, so every site is right-isometric before sampling
// begins -- the property that makes each site's own local squared-norm
// give the correct marginal probability, conditioned on everything
// sampled so far, without ever materializing the full state. Each
// sample_one call is then a read-only O(num_qubits * bond^2) walk: no
// per-shot copy of the state, no allocation beyond one small "running
// left vector" per call.
class MPSCountsSampler : public RegionCountsSampler {
public:
    // measure_targets/bitstring_width follow EngineExecutionRequest's
    // own convention (already-resolved last-write-wins (qubit, clbit)
    // pairs; bitstring_width is the global clbit count).
    MPSCountsSampler(MPSState state,
                     std::span<const std::pair<QubitId, ClbitId>> measure_targets,
                     std::uint32_t bitstring_width);

    [[nodiscard]] BitString sample_one(common::Rng& rng) const override;

private:
    MPSState state_;
    std::vector<std::pair<QubitId, ClbitId>> measure_targets_;
    std::uint32_t bitstring_width_;
};

}  // namespace menos::engines::mps
