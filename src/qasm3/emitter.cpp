// OpenQASM 3 export of a CircuitBuilder (docs/adr/0029 section 3.6): a
// program that menos::qasm3::parse reads back into the identical operations
// and global phase, with each parameter as an `input`.

#include <array>
#include <charconv>
#include <cmath>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <variant>

#include "common/error.h"
#include "ir/gate_table.h"
#include "menos/qasm3.h"
#include "qasm3/stdgates.h"

namespace menos::qasm3 {
namespace {

// Names an exported program may not declare: OpenQASM 3 keywords and
// statement words, the importer's built-in constants and functions, its
// built-in gates, and the `rzz` definition the exporter itself writes.
bool is_reserved(std::string_view name) {
    static const std::set<std::string_view> kReserved = {
        "OPENQASM", "include",  "qubit",    "bit",     "qreg",    "creg",     "const",
        "input",    "output",   "gate",     "measure", "barrier", "reset",    "U",
        "gphase",   "inv",      "pow",      "ctrl",    "negctrl", "if",       "else",
        "for",      "while",    "switch",   "case",    "default", "break",    "continue",
        "end",      "return",   "def",      "extern",  "delay",   "box",      "duration",
        "stretch",  "defcal",   "defcalgrammar",       "cal",     "let",      "opaque",
        "int",      "uint",     "float",    "angle",   "bool",    "complex",  "array",
        "true",     "false",    "pi",       "tau",     "euler",   "sin",      "cos",
        "tan",      "arcsin",   "arccos",   "arctan",  "exp",     "ln",       "sqrt",
        "in",       "durationof", "sizeof", "mod",     "popcount", "rotl",    "rotr",
        "real",     "imag",     "void",     "readonly", "mutable", "rzz",
    };
    return kReserved.contains(name) || detail::find_stdgate(name) != nullptr;
}

std::string number(double value, std::string_view where) {
    if (!std::isfinite(value)) {
        throw common::InvalidCircuit("qasm3 export: " + std::string(where) +
                                     " is not finite and cannot be written");
    }
    std::array<char, 64> buffer{};
    const auto [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    common::ensure(ec == std::errc{}, "qasm3 export: to_chars failed");
    return {buffer.data(), end};
}

// The expression in evaluation order (ParameterExpression::evaluate):
// c1*t1 + c2*t2 + ... + c0, so the importer's left-to-right sum is the same
// sequence of operations.
std::string expression(const ParameterExpression& e, std::string_view where) {
    std::string out;
    for (const ParameterExpression::Term& term : e.terms()) {
        if (!out.empty()) {
            out += " + ";
        }
        if (term.coefficient == 1.0) {
            out += term.name;
        } else {
            out += "(" + number(term.coefficient, where) + ")*" + term.name;
        }
    }
    if (e.constant().has_value()) {
        const std::string c = number(*e.constant(), where);
        out += out.empty() ? c : " + (" + c + ")";
    }
    return out;
}

class Emitter {
public:
    explicit Emitter(const CircuitBuilder& builder) : builder_(builder) {}

    std::string run() {
        check_names();
        std::string body;
        bool uses_rzz = false;
        const auto& ops = builder_.operations();
        for (std::size_t i = 0; i < ops.size(); ++i) {
            const std::string where = "operation " + std::to_string(i);
            std::visit(
                [&](const auto& op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, BuilderGate>) {
                        uses_rzz = uses_rzz || op.opcode == Opcode::RZZ;
                        body += gate(op, where);
                    } else if constexpr (std::is_same_v<T, MeasureOp>) {
                        if (op.readout_error.has_value()) {
                            refuse(where, "a measurement with a readout error");
                        }
                        body += clbit(op.clbit) + " = measure " + qubit(op.qubit) + ";\n";
                    } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                        refuse(where, "an explicit unitary");
                    } else {
                        refuse(where, "a noise channel");
                    }
                },
                ops[i]);
        }

        std::string out = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
        if (uses_rzz) {
            // Read back as one RZZ by the importer (docs/adr/0029 section 8).
            out += "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n";
        }
        for (const std::string& name : builder_.parameters()) {
            out += "input float[64] " + name + ";\n";
        }
        for (const QuantumRegister& r : builder_.qregs()) {
            out += "qubit[" + std::to_string(r.size) + "] " + r.name + ";\n";
        }
        for (const ClassicalRegister& r : builder_.cregs()) {
            out += "bit[" + std::to_string(r.size) + "] " + r.name + ";\n";
        }
        if (builder_.global_phase != 0.0) {
            out += "gphase(" + number(builder_.global_phase, "the global phase") + ");\n";
        }
        return out + body;
    }

private:
    [[noreturn]] static void refuse(const std::string& where, std::string_view what) {
        throw common::UnsupportedFeature("qasm3 export: " + where + " is " + std::string(what) +
                                         ", which OpenQASM 3 export does not support");
    }

    void check_names() const {
        std::set<std::string> seen;
        const auto check = [&](const std::string& name, std::string_view kind) {
            if (is_reserved(name)) {
                throw common::UnsupportedFeature("qasm3 export: " + std::string(kind) + " name '" +
                                                 name +
                                                 "' is an OpenQASM 3 reserved word or gate name");
            }
            if (!seen.insert(name).second) {
                throw common::UnsupportedFeature("qasm3 export: the name '" + name +
                                                 "' is used by more than one register or parameter");
            }
        };
        for (const QuantumRegister& r : builder_.qregs()) {
            check(r.name, "register");
        }
        for (const ClassicalRegister& r : builder_.cregs()) {
            check(r.name, "register");
        }
        for (const std::string& name : builder_.parameters()) {
            check(name, "parameter");
        }
    }

    std::string qubit(QubitId q) const {
        for (const QuantumRegister& r : builder_.qregs()) {
            if (q >= r.offset && q - r.offset < r.size) {
                return r.name + "[" + std::to_string(q - r.offset) + "]";
            }
        }
        throw common::InvalidCircuit("qasm3 export: qubit " + std::to_string(q) +
                                     " is in no register");
    }

    std::string clbit(ClbitId c) const {
        for (const ClassicalRegister& r : builder_.cregs()) {
            if (c >= r.offset && c - r.offset < r.size) {
                return r.name + "[" + std::to_string(c - r.offset) + "]";
            }
        }
        throw common::InvalidCircuit("qasm3 export: classical bit " + std::to_string(c) +
                                     " is in no register");
    }

    std::string gate(const BuilderGate& g, const std::string& where) const {
        std::string name;
        if (g.opcode == Opcode::U) {
            name = "U";
        } else if (g.opcode == Opcode::SXdg) {
            name = "inv @ sx";
        } else {
            name = std::string(ir::gate_info(g.opcode).name);
        }
        std::string out = name;
        if (!g.params.empty()) {
            out += "(";
            for (std::size_t i = 0; i < g.params.size(); ++i) {
                out += (i == 0 ? "" : ", ") + expression(g.params[i], where);
            }
            out += ")";
        }
        for (std::size_t i = 0; i < g.qubits.size(); ++i) {
            out += (i == 0 ? " " : ", ") + qubit(g.qubits[i]);
        }
        return out + ";\n";
    }

    const CircuitBuilder& builder_;
};

}  // namespace

std::string emit(const CircuitBuilder& builder) { return Emitter(builder).run(); }

std::string emit(const Circuit& circuit) { return emit(CircuitBuilder::from_circuit(circuit)); }

}  // namespace menos::qasm3
