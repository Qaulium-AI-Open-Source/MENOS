#pragma once

#include <cstdint>
#include <vector>

#include "menos/simulation_engine.h"

namespace menos::engines::mps {

// What MPSEngine::execute() will allocate, found by replaying the request's
// operations on bond dimensions alone (Phase 21, docs/adr/0025 section 3.4 and
// Q7). The replay follows apply_gate()'s and MPSState's own dimension
// arithmetic step by step -- the SWAP network, every orthogonality-center
// shift, every split -- but holds no amplitudes, so it costs O(operations x
// span) integer steps.
//
// Under AccuracyPolicy::Exact every split keeps min(m, n) singular values,
// numerically zero ones included, so the replayed bonds are exactly the ones
// execute() builds. They can exceed the Schmidt-rank bound
// (analysis::compute_schmidt_rank_bound), which is why the estimate no longer
// uses that bound: measured in docs/adr/0025 section 10.1. Under Controlled a
// split keeps at most min(m, n) values, capped at max_bond_dimension when set;
// the tolerance can only keep fewer, so the replay is then an upper bound.
struct MemoryModel {
    // The most heap bytes execute() holds at once: site tensors at their
    // allocated capacity, every gate's and split's transient buffers
    // (theta, the SVD's input, outputs and workspace, the new site and the
    // next remainder), the sampler's center sweep and per-shot buffers, the
    // counts histogram, and one complex per observable term.
    std::uint64_t peak_bytes = 0;
    // The largest bond any split or shift produced, at least 1.
    std::uint64_t max_bond = 1;
    // Bond i (between sites i and i + 1) after the last operation, before
    // sampling moves the center; empty for one qubit.
    std::vector<std::uint64_t> final_bonds;
};

// Throws common::ResourceExceeded when a figure overflows 64 bits.
[[nodiscard]] MemoryModel model_execution(const EngineEstimateRequest& request);

}  // namespace menos::engines::mps
