#pragma once

#include <vector>

#include "menos/simulation_engine.h"

namespace menos::plan {

// Every engine implementation this build can EXECUTE with, across all
// backends (Phase 10, docs/adr/0014).
//
// Distinct from available_engines(), and the distinction is the whole point:
//
//   available_engines()  is the ALGORITHM list. The planner enumerates its
//                        candidates from it, so it holds exactly one
//                        implementation per Engine and stays CPU-only. Adding
//                        the CUDA engine there would enumerate a duplicate
//                        Dense candidate with identical work units, which is
//                        not what a second backend means.
//
//   execution_engines()  is the RESOLUTION list. It holds every (Engine,
//                        Backend, Tier) triple, so the runtime can map a
//                        Region's triple to exactly one implementation (the
//                        tier is Phase 23's coordinate, docs/adr/0027; every
//                        pre-Phase-23 engine is Tier::InMemory).
//
// The CUDA engine appears here only when the build has CUDA support AND a
// usable device is present at runtime.
[[nodiscard]] const std::vector<const SimulationEngine*>& execution_engines();

// The implementation for an (engine, backend, tier) triple, or nullptr if this
// build has none. Returning nullptr rather than throwing lets the planner ask
// "is there a CUDA dense engine?" without exception control flow; the runtime,
// which requires one, turns a nullptr into a typed error itself. The tier
// defaults to InMemory, which is the implementation every caller before
// Phase 23 asked for.
[[nodiscard]] const SimulationEngine* find_execution_engine(Engine engine,
                                                            Backend backend,
                                                            Tier tier = Tier::InMemory);

}  // namespace menos::plan
