// docs/BUILD_PLAN.md Phase 7: `menos explain`/`menos run`. Linked only
// to the public API (menos::run/preview, menos::qasm2::parse,
// menos::qasm3::parse) -- no engine or planner internals included directly.

#include "cli_impl.h"

#include "menos/menos.h"
#include "menos/qasm2.h"
#include "menos/qasm3.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "common/error.h"

namespace menos::cli {
namespace {

// 2^10 = 1024 amplitudes prints as a readable terminal table; the CLI
// is a small-circuit inspection tool, not a data-export path (spec's
// own framing), and no other precedent in the codebase fixes a value
// for this, so it is CLI-local and display-only -- it does not change
// any engine's own Statevector eligibility.
constexpr std::uint32_t kCliStatevectorQubitLimit = 10;

class UsageError : public std::runtime_error {
public:
    explicit UsageError(const std::string& message) : std::runtime_error(message) {}
};

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw UsageError("cannot open file: " + path);
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

struct Flags {
    std::optional<std::uint64_t> shots;
    std::optional<std::uint64_t> seed;
    bool statevector = false;
    // Phase 17 (docs/adr/0021 section 4): RunOptions::time_budget_seconds.
    std::optional<double> time_budget;
    // Phase 17 (docs/adr/0021 section 2, owner Q2): RunOptions::backend.
    std::optional<menos::Backend> backend;
    // Phase 23 (docs/adr/0027): RunOptions::tier and RunOptions::scratch_directory.
    std::optional<menos::Tier> tier;
    // Phase 24 item 7: RunOptions::fuse_gates; --fusion on|off, on by default.
    bool fuse_gates = true;
    std::optional<std::filesystem::path> scratch_directory;
    // Phase 19 (docs/adr/0023 section 3): values for an OpenQASM 3 program's
    // `input` declarations.
    menos::qasm3::InputBindings inputs;
};

std::uint64_t parse_uint64_flag(const std::string& flag_name, const std::string& value) {
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() || parsed < 0) {
            throw std::invalid_argument("not a non-negative integer");
        }
        return static_cast<std::uint64_t>(parsed);
    } catch (const std::exception&) {
        throw UsageError(flag_name + " requires a non-negative integer argument, got '" + value +
                         "'");
    }
}

// A finite number > 0, the only budgets RunOptions accepts (src/api/menos.cpp).
// Checked here as well so a bad value is a usage error naming the flag, as
// --shots is, rather than an error from the library.
double parse_positive_seconds_flag(const std::string& flag_name, const std::string& value) {
    double parsed = 0.0;
    try {
        std::size_t consumed = 0;
        parsed = std::stod(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (const std::exception&) {
        throw UsageError(flag_name + " requires a number of seconds, got '" + value + "'");
    }
    if (!std::isfinite(parsed) || parsed <= 0.0) {
        throw UsageError(flag_name + " requires a finite number of seconds greater than 0, got '" +
                         value + "'");
    }
    return parsed;
}

Flags parse_flags(const std::vector<std::string>& args, std::size_t start, bool allow_statevector) {
    Flags flags;
    for (std::size_t i = start; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--shots") {
            if (i + 1 >= args.size()) {
                throw UsageError("--shots requires an argument");
            }
            flags.shots = parse_uint64_flag("--shots", args[++i]);
        } else if (arg == "--seed") {
            if (i + 1 >= args.size()) {
                throw UsageError("--seed requires an argument");
            }
            flags.seed = parse_uint64_flag("--seed", args[++i]);
        } else if (arg == "--time-budget") {
            if (i + 1 >= args.size()) {
                throw UsageError("--time-budget requires an argument");
            }
            flags.time_budget = parse_positive_seconds_flag("--time-budget", args[++i]);
        } else if (arg == "--backend") {
            if (i + 1 >= args.size()) {
                throw UsageError("--backend requires an argument");
            }
            const std::string& value = args[++i];
            if (value == "cpu") {
                flags.backend = menos::Backend::Cpu;
            } else if (value == "cuda") {
                flags.backend = menos::Backend::Cuda;
            } else {
                throw UsageError("--backend must be 'cpu' or 'cuda', got '" + value + "'");
            }
        } else if (arg == "--tier") {
            if (i + 1 >= args.size()) {
                throw UsageError("--tier requires an argument");
            }
            const std::string& value = args[++i];
            if (value == "disk") {
                flags.tier = menos::Tier::Disk;
            } else if (value == "in_memory") {
                flags.tier = menos::Tier::InMemory;
            } else {
                throw UsageError("--tier must be 'disk' or 'in_memory', got '" + value + "'");
            }
        } else if (arg == "--fusion") {
            if (i + 1 >= args.size()) {
                throw UsageError("--fusion requires an argument");
            }
            const std::string& value = args[++i];
            if (value == "on") {
                flags.fuse_gates = true;
            } else if (value == "off") {
                flags.fuse_gates = false;
            } else {
                throw UsageError("--fusion must be 'on' or 'off', got '" + value + "'");
            }
        } else if (arg == "--scratch-dir") {
            if (i + 1 >= args.size()) {
                throw UsageError("--scratch-dir requires an argument");
            }
            flags.scratch_directory = std::filesystem::path(args[++i]);
        } else if (arg == "--input") {
            if (i + 1 >= args.size()) {
                throw UsageError("--input requires an argument NAME=VALUE");
            }
            const std::string& binding = args[++i];
            const std::size_t equals = binding.find('=');
            if (equals == std::string::npos || equals == 0) {
                throw UsageError("--input requires NAME=VALUE, got '" + binding + "'");
            }
            const std::string name = binding.substr(0, equals);
            const std::string value = binding.substr(equals + 1);
            double parsed = 0.0;
            try {
                std::size_t consumed = 0;
                parsed = std::stod(value, &consumed);
                if (consumed != value.size()) {
                    throw std::invalid_argument("trailing characters");
                }
            } catch (const std::exception&) {
                throw UsageError(
                    std::format("--input {} requires a number, got '{}'", name, value));
            }
            if (!std::isfinite(parsed)) {
                throw UsageError(
                    std::format("--input {} requires a finite number, got '{}'", name, value));
            }
            if (!flags.inputs.emplace(name, parsed).second) {
                throw UsageError("--input " + name + " is given more than once");
            }
        } else if (arg == "--statevector" && allow_statevector) {
            flags.statevector = true;
        } else {
            throw UsageError("unrecognized argument: " + arg);
        }
    }
    return flags;
}

// The importer is chosen from the program's header (docs/adr/0023 section 9,
// Q6): `OPENQASM 2...` goes to qasm2::parse, `OPENQASM 3...` to qasm3::parse,
// each of which checks the exact version itself. A program without a header
// is refused rather than guessed at.
menos::Circuit parse_program(const std::string& text, const menos::qasm3::InputBindings& inputs) {
    std::size_t i = 0;
    for (;;) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        if (text.compare(i, 2, "//") == 0) {
            i = text.find('\n', i);
            i = i == std::string::npos ? text.size() : i;
        } else if (text.compare(i, 2, "/*") == 0) {
            const std::size_t end = text.find("*/", i + 2);
            i = end == std::string::npos ? text.size() : end + 2;
        } else {
            break;
        }
    }
    const std::string keyword = "OPENQASM";
    if (text.compare(i, keyword.size(), keyword) != 0) {
        throw UsageError(
            "the program has no 'OPENQASM 2.0;' or 'OPENQASM 3.0;' header; add one "
            "so its language version is known");
    }
    i += keyword.size();
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
        ++i;
    }
    if (text.compare(i, 1, "2") == 0) {
        if (!inputs.empty()) {
            throw UsageError("--input applies only to OpenQASM 3 programs");
        }
        return menos::qasm2::parse(text);
    }
    if (text.compare(i, 1, "3") == 0) {
        return menos::qasm3::parse(text, inputs);
    }
    throw UsageError("unsupported OpenQASM version in the header; expected 2.0 or 3");
}

// Highest classical bit first (matching the Qiskit adapter's own
// convention -- Qiskit's own Counts/postprocess machinery performs
// this same formatting there, since the adapter itself never formats a
// binary string, docs/adr/0004), implemented independently here per
// CLAUDE.md's rule restricting this formatting to specific, named
// consumers.
std::string format_counts_key(const menos::BitString& key, std::uint32_t num_clbits) {
    std::string s(num_clbits, '0');
    for (std::uint32_t bit = 0; bit < num_clbits; ++bit) {
        if (key.test(bit)) {
            s[num_clbits - 1 - bit] = '1';
        }
    }
    return s;
}

int cmd_explain(const std::string& file, const Flags& flags, std::ostream& out) {
    const menos::Circuit circuit = parse_program(read_file(file), flags.inputs);
    // explain() never executes, so the concrete shot count has no
    // effect on the plan (src/plan/planner.cpp only ever tests
    // shots > 0); default to 1 purely to request the counts output
    // mode when --shots is omitted, since that is the only output mode
    // this command's flag set can express and planning needs some
    // request to plan for. seed is not consumed by planning at all
    // (only by Runtime::execute's sampling), so it defaults to 0 when
    // omitted regardless.
    const menos::RunOptions options{.shots = flags.shots.value_or(1),
                                    .want_statevector = false,
                                    .seed = flags.seed.value_or(0),
                                    .time_budget_seconds = flags.time_budget,
                                    .fuse_gates = flags.fuse_gates,
                                    .backend = flags.backend,
                                    .tier = flags.tier,
                                    .scratch_directory = flags.scratch_directory};
    const menos::Plan plan = menos::preview(circuit, options);
    out << plan.explain() << "\n";
    return 0;
}

int cmd_run(const std::string& file, const Flags& flags, std::ostream& out) {
    if (!flags.shots.has_value()) {
        throw UsageError("--shots is required for 'run'");
    }
    if (!flags.seed.has_value()) {
        throw UsageError("--seed is required for 'run'");
    }
    const menos::Circuit circuit = parse_program(read_file(file), flags.inputs);
    if (flags.statevector && circuit.num_qubits() > kCliStatevectorQubitLimit) {
        throw UsageError(
            std::format("--statevector requested for a {}-qubit circuit, above this CLI's {}-qubit "
                        "inspection limit; use the Python API for larger circuits",
                        circuit.num_qubits(),
                        kCliStatevectorQubitLimit));
    }

    const menos::RunOptions options{.shots = *flags.shots,
                                    .want_statevector = flags.statevector,
                                    .seed = *flags.seed,
                                    .time_budget_seconds = flags.time_budget,
                                    .fuse_gates = flags.fuse_gates,
                                    .backend = flags.backend,
                                    .tier = flags.tier,
                                    .scratch_directory = flags.scratch_directory};
    const menos::Result result = menos::run(circuit, options);
    out << result.explain() << "\n";

    if (!result.counts().empty()) {
        const std::uint32_t num_clbits = circuit.num_clbits();
        for (const auto& [key, count] : result.counts()) {
            out << format_counts_key(key, num_clbits) << ": " << count << "\n";
        }
    }
    if (result.statevector().has_value()) {
        const auto& amplitudes = *result.statevector();
        for (std::size_t i = 0; i < amplitudes.size(); ++i) {
            out << i << ": "
                << std::format("{:.12g}{:+.12g}i", amplitudes[i].real(), amplitudes[i].imag())
                << "\n";
        }
    }
    return 0;
}

int dispatch(const std::vector<std::string>& args, std::ostream& out) {
    if (args.size() < 2) {
        throw UsageError(
            "usage: menos <explain|run> <file.qasm> [--shots N] [--seed N] [--statevector] "
            "[--time-budget SECONDS] [--backend cpu|cuda] [--tier disk|in_memory] "
            "[--scratch-dir PATH] [--fusion on|off] [--input NAME=VALUE]...");
    }
    const std::string& command = args[0];
    const std::string& file = args[1];
    if (command == "explain") {
        const Flags flags = parse_flags(args, 2, /*allow_statevector=*/false);
        return cmd_explain(file, flags, out);
    }
    if (command == "run") {
        const Flags flags = parse_flags(args, 2, /*allow_statevector=*/true);
        return cmd_run(file, flags, out);
    }
    throw UsageError("unknown command '" + command + "' -- expected 'explain' or 'run'");
}

}  // namespace

// out/err are never swapped at either call site (main.cpp passes
// std::cout/std::cerr in that fixed order; tests/cli/golden_test.cpp's
// own two std::ostringstreams are always constructed and named out/err
// in that same order) -- NOLINT rather than a wrapper type for two
// plain output streams.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    try {
        return dispatch(args, out);
    } catch (const UsageError& e) {
        err << "error: " << e.what() << "\n";
        return 1;
    } catch (const menos::common::Error& e) {
        err << "error: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace menos::cli
