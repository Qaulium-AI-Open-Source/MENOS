#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "common/rng.h"
#include "menos/bitstring.h"
#include "menos/circuit.h"

namespace menos::engines::noise {

// Readout errors, shared by every engine that samples a noisy job's counts:
// the density-matrix engine (Phase 12, docs/adr/0016 section 4) and the
// trajectory engine (Phase 13, docs/adr/0017 section 2). Keeping one copy is
// what makes the two engines' recorded outcomes follow the same rule, draw for
// draw.

// The readout error of one measure target that has one.
struct ReadoutTarget {
    ClbitId clbit;
    ReadoutError error;
};

// The readout error of each measure target that has one, in measure_targets
// order. A target is the surviving (last-write-wins) measurement of its clbit,
// and the runtime only gives a region a target whose surviving write is in that
// region, so it is the region's last MeasureOp writing that clbit.
[[nodiscard]] std::vector<ReadoutTarget> readout_targets(
    std::span<const Operation> operations,
    std::span<const std::pair<QubitId, ClbitId>> measure_targets);

// The heap bytes readout_targets() returns for up to num_targets targets: it
// grows its vector one target at a time (docs/adr/0025 section 3.4).
[[nodiscard]] std::uint64_t readout_targets_bytes(std::uint64_t num_targets);

// Records each readout-errored bit of a true outcome through its assignment
// matrix: one uniform per target, in order; the bit flips when the uniform is
// below P(recorded != true | true).
void apply_readout(BitString& outcome, std::span<const ReadoutTarget> targets, common::Rng& rng);

}  // namespace menos::engines::noise
