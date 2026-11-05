#pragma once

#include <string_view>

#include "menos/plan.h"

namespace menos::plan {

// The report spellings of the plan enums, defined once in plan_impl.cpp.
//
// These were local to Plan::explain() until Phase 11 gave Result::explain() a
// section of its own that also names an engine and a backend (the replan
// report, docs/adr/0015). Two copies of the same switch would be free to drift,
// and a report that spelled the same engine differently in two of its own
// sections would be worse than useless. Internal, not public API: nothing
// outside src/ needs to spell an enum.
[[nodiscard]] std::string_view strategy_name(Strategy strategy);
[[nodiscard]] std::string_view engine_name(Engine engine);
[[nodiscard]] std::string_view backend_name(Backend backend);
[[nodiscard]] std::string_view tier_name(Tier tier);

}  // namespace menos::plan
