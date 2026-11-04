// "The same plan, up to pruning" (Phase 21 section D, docs/adr/0025 section 13).
//
// The owner's invariant for section D: the chosen plan and its cost stay
// identical, and explain() may change only where a candidate, or an engine in
// a Factorized region's "why", is now reported as pruned by branch-and-bound
// instead of costed in full. This compares a reference outcome (the frozen
// d89dde3 planner, or a golden file recorded from it) with a live one
// (planner_corpus::outcome_of) under exactly that rule:
//
// - both outcomes have the same lines, in the same order;
// - a line that differs must be either
//   - a candidate line, same index and strategy, that the live planner reports
//     as pruned, and that was not the reference winner. If the reference
//     costed it as feasible, its work units must be strictly above the
//     winner's: a candidate can be pruned only once a cheaper feasible one
//     exists (ties go to the earlier candidate, and pruning needs strictly
//     more); or
//   - a region line whose fields before `why=` are identical and whose "why"
//     differs only in engines the live planner reports as pruned: such an
//     engine may be absent from the reference (it was feasible but costlier
//     there) or carry any reason;
// - everything else, including the strategy, the estimate, every region's
//   fields and every refusal, is byte-identical.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace menos::planner_corpus {

namespace prune_equivalence_detail {

inline constexpr std::string_view kPruned = "pruned by branch-and-bound";

inline std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        lines.push_back(line);
    }
    return lines;
}

// The value of `field=` up to the next space, or nullopt.
inline std::optional<std::string> field(const std::string& line, std::string_view name) {
    const std::string key = " " + std::string(name) + "=";
    const std::size_t at = line.find(key);
    if (at == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t begin = at + key.size();
    return line.substr(begin, line.find(' ', begin) - begin);
}

inline std::optional<std::uint64_t> number(const std::string& line, std::string_view name) {
    const std::optional<std::string> value = field(line, name);
    if (!value.has_value() || value->empty()) {
        return std::nullopt;
    }
    return std::stoull(*value);
}

// "candidate 4strategy=direct_tensor_network" -- the part before " feasible=".
inline std::string candidate_head(const std::string& line) {
    return line.substr(0, line.find(" feasible="));
}

struct Why {
    std::string winner;
    std::map<std::string, std::string> rejected;  // engine -> reason
};

// Parses `why="<winner> (<engine> rejected: <reason>)..."`. Engine names are
// the fixed set SimulationEngine::name() returns.
inline std::optional<Why> parse_why(const std::string& why) {
    static const std::vector<std::string> kEngines{"dense",
                                                   "dense_cuda",
                                                   "stabilizer",
                                                   "pauli",
                                                   "mps",
                                                   "tensor_network",
                                                   "density_matrix",
                                                   "trajectory"};
    // Every " (<engine> rejected: " marker, in order of position.
    std::map<std::size_t, std::string> markers;
    for (const std::string& engine : kEngines) {
        const std::string marker = " (" + engine + " rejected: ";
        for (std::size_t at = why.find(marker); at != std::string::npos;
             at = why.find(marker, at + 1)) {
            markers[at] = engine;
        }
    }
    Why out;
    out.winner = why.substr(0, markers.empty() ? why.size() : markers.begin()->first);
    for (auto it = markers.begin(); it != markers.end(); ++it) {
        const std::size_t begin =
            it->first + 2 + it->second.size() + 11;  // " (" name " rejected: "
        const auto next = std::next(it);
        const std::size_t end = next == markers.end() ? why.size() : next->first;
        if (end <= begin || why[end - 1] != ')') {
            return std::nullopt;
        }
        if (!out.rejected.emplace(it->second, why.substr(begin, end - 1 - begin)).second) {
            return std::nullopt;  // one engine listed twice
        }
    }
    return out;
}

inline std::optional<std::string> compare_region(const std::string& reference,
                                                 const std::string& live) {
    const std::string marker = " why=\"";
    const std::size_t ref_at = reference.find(marker);
    const std::size_t live_at = live.find(marker);
    if (live_at == std::string::npos || reference.substr(0, ref_at) != live.substr(0, live_at)) {
        return "region fields differ";
    }
    if (!live.ends_with("\"") || (ref_at != std::string::npos && !reference.ends_with("\""))) {
        return "region why is not quoted";
    }
    const std::string live_why =
        live.substr(live_at + marker.size(), live.size() - live_at - marker.size() - 1);
    const std::string ref_why =
        ref_at == std::string::npos
            ? std::string()
            : reference.substr(ref_at + marker.size(),
                               reference.size() - ref_at - marker.size() - 1);
    const std::optional<Why> l = parse_why(live_why);
    const std::optional<Why> r = parse_why(ref_why);
    if (!l.has_value() || !r.has_value()) {
        return "region why does not parse";
    }
    if (l->winner != r->winner) {
        return "region engine differs";
    }
    for (const auto& [engine, reason] : l->rejected) {
        if (reason.starts_with(kPruned)) {
            continue;  // may be absent from the reference or carry any reason
        }
        const auto found = r->rejected.find(engine);
        if (found == r->rejected.end() || found->second != reason) {
            return "region rejection of " + engine + " differs";
        }
    }
    // A reference rejection must still be listed live: with the same reason
    // (checked above) or as pruned.
    for (const auto& entry : r->rejected) {
        if (!l->rejected.contains(entry.first)) {
            return "region rejection of " + entry.first + " missing from the live plan";
        }
    }
    return std::nullopt;
}

}  // namespace prune_equivalence_detail

// nullopt when `live` is `reference` up to pruning (the rule above); otherwise
// a description of the first difference that is not.
inline std::optional<std::string> differs_beyond_pruning(const std::string& reference,
                                                         const std::string& live) {
    namespace d = prune_equivalence_detail;
    if (reference == live) {
        return std::nullopt;
    }
    if (reference.starts_with("THROWS ") || live.starts_with("THROWS ")) {
        return "the outcome is a refusal on one side and differs";
    }
    const std::vector<std::string> ref = d::lines_of(reference);
    const std::vector<std::string> now = d::lines_of(live);
    if (ref.size() != now.size()) {
        return "different line counts";
    }
    // The winner's work units: the "estimated" line, which must itself match.
    std::optional<std::uint64_t> winner_work;
    for (const std::string& line : ref) {
        if (line.starts_with("estimated ")) {
            winner_work = d::number(line, "work_units");
        }
    }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const std::string& r = ref[i];
        const std::string& l = now[i];
        if (r == l) {
            continue;
        }
        const std::string where = "line " + std::to_string(i + 1) + ": ";
        if (r.starts_with("candidate ") && l.starts_with("candidate ")) {
            if (d::candidate_head(r) != d::candidate_head(l)) {
                return where + "candidate order differs";
            }
            if (l.find(" feasible=false winner=false reason=\"" + std::string(d::kPruned)) ==
                std::string::npos) {
                return where + "candidate differs and is not pruned in the live plan";
            }
            if (r.find(" winner=true") != std::string::npos) {
                return where + "the reference winner is pruned in the live plan";
            }
            if (d::field(r, "feasible") == "true") {
                const std::optional<std::uint64_t> work = d::number(r, "work_units");
                if (!work.has_value() || !winner_work.has_value() || !(*work > *winner_work)) {
                    return where + "a pruned candidate was not strictly costlier than the winner";
                }
            }
            continue;
        }
        if (r.starts_with("region ") && l.starts_with("region ")) {
            if (const std::optional<std::string> why = d::compare_region(r, l)) {
                return where + *why;
            }
            continue;
        }
        return where + "differs";
    }
    return std::nullopt;
}

}  // namespace menos::planner_corpus
