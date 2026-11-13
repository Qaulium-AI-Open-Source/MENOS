#include "menos/qasm3.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "common/error.h"
#include "qasm3/lexer.h"
#include "qasm3/modifiers.h"
#include "qasm3/stdgates.h"

namespace menos::qasm3 {
namespace {

using common::InvalidCircuit;
using common::UnsupportedFeature;
using detail::GlobalPhase;
using detail::Instr;
using detail::Token;
using detail::TokenKind;

template <typename V>
using NameMap = std::map<std::string, V, std::less<>>;

// Statements outside the static-circuit subset, refused by their keyword
// (docs/adr/0023 section 2.2).
const NameMap<std::string_view>& refused_keywords() {
    static const NameMap<std::string_view> kRefused = {
        {"reset", "'reset' is not supported (deferred to dynamic circuits)"},
        {"if", "classical control flow ('if') is not supported"},
        {"else", "classical control flow ('else') is not supported"},
        {"for", "classical control flow ('for') is not supported"},
        {"while", "classical control flow ('while') is not supported"},
        {"switch", "classical control flow ('switch') is not supported"},
        {"break", "classical control flow ('break') is not supported"},
        {"continue", "classical control flow ('continue') is not supported"},
        {"end", "'end' is not supported"},
        {"return", "subroutines ('return') are not supported"},
        {"def", "subroutines ('def') are not supported"},
        {"extern", "'extern' declarations are not supported"},
        {"delay", "timing ('delay') is not supported"},
        {"box", "timing ('box') is not supported"},
        {"duration", "timing ('duration') is not supported"},
        {"stretch", "timing ('stretch') is not supported"},
        {"defcal", "calibration grammar ('defcal') is not supported"},
        {"defcalgrammar", "calibration grammar ('defcalgrammar') is not supported"},
        {"cal", "calibration grammar ('cal') is not supported"},
        {"output", "'output' declarations are not supported"},
        {"let", "aliasing ('let') is not supported"},
        {"opaque", "'opaque' gate declarations are not supported"},
        {"int", "classical variables are not supported (only 'const' and 'input')"},
        {"uint", "classical variables are not supported (only 'const' and 'input')"},
        {"float", "classical variables are not supported (only 'const' and 'input')"},
        {"angle", "classical variables are not supported (only 'const' and 'input')"},
        {"bool", "classical variables are not supported (only 'const' and 'input')"},
        {"complex", "classical variables are not supported (only 'const' and 'input')"},
        {"array", "arrays are not supported"},
    };
    return kRefused;
}

bool is_modifier(std::string_view text) {
    return text == "inv" || text == "pow" || text == "ctrl" || text == "negctrl";
}

// A value of a constant expression; is_int tracks OpenQASM's integer type so
// integer division, whose rounding this subset does not define, is refused.
struct Value {
    double value;
    bool is_int;
};

struct Register {
    std::uint32_t offset;
    std::uint32_t size;
    bool sized;  // `qubit[n] q` / `qreg q[n]`, as opposed to a single `qubit q`
};

struct GateDef {
    std::size_t order;  // definition order; a body may call only earlier gates
    std::vector<std::string> params;
    std::vector<std::string> qubits;
    std::size_t body_begin;  // token index after '{'
    std::size_t body_end;    // token index of the matching '}'
    // `gate rzz(t) a, b { cx a, b; rz(t) b; cx a, b; }` exactly, with
    // stdgates.inc included: read as one RZZ, which equals that body exactly
    // (docs/adr/0029 section 8). Any other rzz body is expanded.
    bool canonical_rzz = false;
};

// The names visible inside a gate body while it is expanded.
struct Scope {
    const NameMap<double>* params;
    const NameMap<QubitId>* qubits;
    std::size_t gate_order;  // the gate being expanded
};

struct Modifier {
    enum class Kind : std::uint8_t { Inv, Pow, Ctrl, NegCtrl } kind;
    std::int64_t count;  // exponent for Pow, number of controls for Ctrl/NegCtrl
};

class Parser {
public:
    Parser(std::vector<Token> tokens, const InputBindings& inputs)
        : tokens_(std::move(tokens)), inputs_(inputs) {}

    Circuit run() {
        if (at_ident("OPENQASM")) {
            parse_version();
        }
        while (peek().kind != TokenKind::Eof) {
            parse_statement();
        }
        if (num_qubits_ == 0) {
            throw InvalidCircuit("qasm3: no qubit declaration found");
        }
        for (const auto& [name, value] : inputs_) {
            if (!used_inputs_.contains(name)) {
                throw InvalidCircuit("qasm3: the binding for '" + name +
                                     "' names no 'input' declaration");
            }
        }
        Circuit circuit(num_qubits_, num_clbits_);
        for (Operation& op : operations_) {
            std::visit(
                [&circuit](auto& o) {
                    using T = std::decay_t<decltype(o)>;
                    if constexpr (std::is_same_v<T, GateOp>) {
                        circuit.append(o.opcode, std::move(o.qubits), std::move(o.params));
                    } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                        circuit.append_unitary(std::move(o.qubits), std::move(o.matrix));
                    } else if constexpr (std::is_same_v<T, MeasureOp>) {
                        circuit.append_measure(o.qubit, o.clbit);
                    } else {
                        static_assert(std::is_same_v<T, ChannelOp>);
                        common::ensure(false, "qasm3: the parser produced a noise channel");
                    }
                },
                op);
        }
        circuit.global_phase = global_phase_;
        return circuit;
    }

private:
    // ---- tokens and errors ------------------------------------------------

    const Token& peek(std::size_t ahead = 0) const {
        const std::size_t i = pos_ + ahead;
        return i < tokens_.size() ? tokens_[i] : tokens_.back();
    }
    const Token& advance() {
        return tokens_[pos_ < tokens_.size() - 1 ? pos_++ : pos_];
    }
    bool at_ident(std::string_view text) const {
        return peek().kind == TokenKind::Ident && peek().text == text;
    }

    static std::string location(const Token& token) {
        return " at line " + std::to_string(token.line) + ", column " +
               std::to_string(token.column);
    }
    [[noreturn]] static void fail_malformed(const std::string& what, const Token& token) {
        throw InvalidCircuit("qasm3: " + what + location(token));
    }
    [[noreturn]] static void fail_unsupported(const std::string& what, const Token& token) {
        throw UnsupportedFeature("qasm3: " + what + location(token));
    }

    const Token& expect(TokenKind kind, const std::string& what) {
        if (peek().kind != kind) {
            fail_malformed("expected " + what, peek());
        }
        return advance();
    }
    std::string expect_ident(const std::string& what) {
        return expect(TokenKind::Ident, what).text;
    }

    // ---- names ------------------------------------------------------------

    bool name_taken(std::string_view name) const {
        return qregs_.contains(name) || cregs_.contains(name) || constants_.contains(name) ||
               gates_.contains(name) || name == "U" || name == "gphase" ||
               (stdgates_ && detail::find_stdgate(name) != nullptr) || is_builtin_constant(name);
    }
    void declare(const Token& name_token) {
        if (name_taken(name_token.text)) {
            fail_malformed("redefinition of '" + name_token.text + "'", name_token);
        }
    }

    static bool is_builtin_constant(std::string_view name) {
        return name == "pi" || name == "π" || name == "tau" || name == "τ" || name == "euler" ||
               name == "ℯ";
    }

    // ---- statements -------------------------------------------------------

    void parse_version() {
        const Token& keyword = advance();
        const Token& version = peek();
        if ((version.kind != TokenKind::Int && version.kind != TokenKind::Real) ||
            (version.text != "3" && version.text != "3.0")) {
            fail_unsupported(
                "OpenQASM version '" + version.text + "' is not supported (only 3 and 3.0)",
                keyword);
        }
        advance();
        expect(TokenKind::Semicolon, "';' after the version");
    }

    void parse_statement() {
        const Token& token = peek();
        if (token.kind == TokenKind::At) {
            fail_unsupported("annotations are not supported", token);
        }
        if (token.kind == TokenKind::Hash) {
            fail_unsupported("pragmas are not supported", token);
        }
        if (token.kind != TokenKind::Ident) {
            fail_malformed("expected a statement", token);
        }
        const std::string& word = token.text;
        if (word == "OPENQASM") {
            fail_malformed("the version must be the first statement", token);
        } else if (word == "include") {
            parse_include();
        } else if (word == "qubit" || word == "bit") {
            parse_declaration(word == "qubit");
        } else if (word == "qreg" || word == "creg") {
            parse_old_declaration(word == "qreg");
        } else if (word == "const") {
            parse_const();
        } else if (word == "input") {
            parse_input();
        } else if (word == "gate") {
            parse_gate_definition();
        } else if (word == "measure") {
            parse_measure_arrow();
        } else if (word == "barrier") {
            skip_statement();
        } else if (const auto it = refused_keywords().find(word); it != refused_keywords().end()) {
            fail_unsupported(std::string(it->second), token);
        } else if (peek(1).kind == TokenKind::Equals ||
                   (cregs_.contains(word) && peek(1).kind == TokenKind::LBracket)) {
            parse_measure_assignment();
        } else {
            emit(parse_gate_call(nullptr));
        }
    }

    void skip_statement() {
        while (peek().kind != TokenKind::Semicolon) {
            if (peek().kind == TokenKind::Eof) {
                fail_malformed("expected ';'", peek());
            }
            advance();
        }
        advance();
    }

    void parse_include() {
        const Token& keyword = advance();
        const Token& file = expect(TokenKind::String, "a file name after 'include'");
        expect(TokenKind::Semicolon, "';' after an include");
        if (file.text != "stdgates.inc") {
            fail_unsupported("only 'include \"stdgates.inc\";' is supported", keyword);
        }
        if (stdgates_) {
            fail_malformed("'stdgates.inc' is included twice", keyword);
        }
        for (const auto& [name, def] : gates_) {
            if (detail::find_stdgate(name) != nullptr) {
                fail_malformed("gate '" + name + "' is defined before including stdgates.inc",
                               keyword);
            }
        }
        stdgates_ = true;
    }

    std::uint32_t parse_size() {
        const Token& open = expect(TokenKind::LBracket, "'['");
        const Value size = parse_expr();
        expect(TokenKind::RBracket, "']'");
        if (!size.is_int || size.value < 1 || size.value > 1e9) {
            fail_malformed("a register size must be a positive integer", open);
        }
        return static_cast<std::uint32_t>(size.value);
    }

    void add_register(bool quantum, const Token& name, std::uint32_t size, bool sized) {
        declare(name);
        std::uint32_t& count = quantum ? num_qubits_ : num_clbits_;
        (quantum ? qregs_ : cregs_).emplace(name.text, Register{count, size, sized});
        count += size;
    }

    // qubit[n] q;  qubit q;  bit[n] c;  bit c;
    void parse_declaration(bool quantum) {
        advance();
        const bool sized = peek().kind == TokenKind::LBracket;
        const std::uint32_t size = sized ? parse_size() : 1;
        const Token& name = expect(TokenKind::Ident, "a register name");
        if (peek().kind == TokenKind::Equals) {
            fail_unsupported("initialising a classical bit is not supported", peek());
        }
        expect(TokenKind::Semicolon, "';' after a declaration");
        add_register(quantum, name, size, sized);
    }

    // qreg q[n];  creg c[n];
    void parse_old_declaration(bool quantum) {
        advance();
        const Token& name = expect(TokenKind::Ident, "a register name");
        const bool sized = peek().kind == TokenKind::LBracket;
        const std::uint32_t size = sized ? parse_size() : 1;
        expect(TokenKind::Semicolon, "';' after a declaration");
        add_register(quantum, name, size, sized);
    }

    // The type of a const or input; returns whether it is an integer type.
    // float[64] is IEEE double, which is exact here; other widths and angle[n]
    // would round, so they are refused rather than approximated.
    bool parse_numeric_type(bool allow_int) {
        const Token& type = peek();
        if (type.kind != TokenKind::Ident) {
            fail_malformed("expected a type", type);
        }
        const std::string& name = type.text;
        const bool integer = name == "int" || name == "uint";
        if (!(name == "float" || name == "angle" || (allow_int && integer))) {
            fail_unsupported("a '" + name + "' declaration is not supported", type);
        }
        advance();
        if (peek().kind == TokenKind::LBracket) {
            const Token& open = peek();
            const std::uint32_t width = parse_size();
            if (name == "float" && width != 64) {
                fail_unsupported("float[" + std::to_string(width) +
                                     "] is not supported (only float and float[64])",
                                 open);
            }
            if (name == "angle") {
                fail_unsupported("angle[" + std::to_string(width) +
                                     "] is not supported: a sized angle rounds its value",
                                 open);
            }
        }
        return integer;
    }

    // const float x = expr;  const int n = expr;  const angle a = expr;
    void parse_const() {
        advance();
        const bool integer = parse_numeric_type(true);
        const Token& name = expect(TokenKind::Ident, "a constant name");
        expect(TokenKind::Equals, "'=' in a const declaration");
        const Value value = parse_expr();
        expect(TokenKind::Semicolon, "';' after a const declaration");
        if (integer && (!value.is_int)) {
            fail_malformed("an integer constant needs an integer value", name);
        }
        declare(name);
        constants_.emplace(name.text, value);
    }

    // input float theta;  input float[64] theta;  input angle theta;
    void parse_input() {
        advance();
        static_cast<void>(parse_numeric_type(false));
        const Token& name = expect(TokenKind::Ident, "an input name");
        expect(TokenKind::Semicolon, "';' after an input declaration");
        declare(name);
        const auto binding = inputs_.find(name.text);
        if (binding == inputs_.end()) {
            fail_malformed("input '" + name.text + "' has no binding", name);
        }
        if (!std::isfinite(binding->second)) {
            fail_malformed("the binding for input '" + name.text + "' is not finite", name);
        }
        used_inputs_.insert(name.text);
        constants_.emplace(name.text, Value{binding->second, false});
    }

    // gate name(p, ...) a, b { body }
    void parse_gate_definition() {
        advance();
        const Token& name = expect(TokenKind::Ident, "a gate name");
        declare(name);
        GateDef def;
        def.order = gates_.size();
        if (peek().kind == TokenKind::LParen) {
            advance();
            if (peek().kind != TokenKind::RParen) {
                def.params.push_back(expect_ident("a parameter name"));
                while (peek().kind == TokenKind::Comma) {
                    advance();
                    def.params.push_back(expect_ident("a parameter name"));
                }
            }
            expect(TokenKind::RParen, "')' after gate parameters");
        }
        def.qubits.push_back(expect_ident("a qubit argument"));
        while (peek().kind == TokenKind::Comma) {
            advance();
            def.qubits.push_back(expect_ident("a qubit argument"));
        }
        const std::set<std::string, std::less<>> params(def.params.begin(), def.params.end());
        const std::set<std::string, std::less<>> qubits(def.qubits.begin(), def.qubits.end());
        if (params.size() != def.params.size() || qubits.size() != def.qubits.size()) {
            fail_malformed("gate '" + name.text + "' repeats an argument name", name);
        }
        expect(TokenKind::LBrace, "'{' to open the gate body");
        def.body_begin = pos_;
        while (peek().kind != TokenKind::RBrace) {
            if (peek().kind == TokenKind::Eof || peek().kind == TokenKind::LBrace) {
                fail_malformed("expected '}' to close gate '" + name.text + "'", peek());
            }
            advance();
        }
        def.body_end = pos_;
        advance();
        def.canonical_rzz = is_canonical_rzz(name.text, def);
        gates_.emplace(name.text, std::move(def));
    }

    // Whether def is the circuit builder's exported rzz definition, token for
    // token (docs/adr/0029 section 3.6). Its body calls the stdgates.inc cx and
    // rz, so stdgates.inc must be included (which also stops a user definition
    // of cx or rz).
    bool is_canonical_rzz(std::string_view name, const GateDef& def) const {
        if (name != "rzz" || !stdgates_ || def.params.size() != 1 || def.qubits.size() != 2) {
            return false;
        }
        const std::string& t = def.params[0];
        const std::string& a = def.qubits[0];
        const std::string& b = def.qubits[1];
        const std::array<std::string_view, 16> expected = {
            "cx", a, ",", b, ";", "rz", "(", t, ")", b, ";", "cx", a, ",", b, ";"};
        if (def.body_end - def.body_begin != expected.size()) {
            return false;
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (tokens_[def.body_begin + i].text != expected[i]) {
                return false;
            }
        }
        return true;
    }

    // ---- measurement ------------------------------------------------------

    // One or more (qubit, clbit) pairs: an indexed element, a single unsized
    // qubit/bit, or a whole register.
    std::vector<std::uint32_t> parse_bits(bool quantum) {
        const Token& name = expect(TokenKind::Ident, quantum ? "a qubit" : "a classical bit");
        const NameMap<Register>& regs = quantum ? qregs_ : cregs_;
        const auto it = regs.find(name.text);
        if (it == regs.end()) {
            fail_malformed(std::string(quantum ? "undeclared qubit register '"
                                               : "undeclared classical register '") +
                               name.text + "'",
                           name);
        }
        const Register& reg = it->second;
        if (peek().kind == TokenKind::LBracket) {
            return {reg.offset + parse_index(reg, name)};
        }
        std::vector<std::uint32_t> bits;
        bits.reserve(reg.size);
        for (std::uint32_t i = 0; i < reg.size; ++i) {
            bits.push_back(reg.offset + i);
        }
        return bits;
    }

    std::uint32_t parse_index(const Register& reg, const Token& name) {
        const Token& open = expect(TokenKind::LBracket, "'['");
        const Value index = parse_expr();
        if (peek().kind == TokenKind::Comma ||
            (peek().kind == TokenKind::Other && peek().text == ":")) {
            fail_unsupported("register slices are not supported", peek());
        }
        expect(TokenKind::RBracket, "']'");
        if (!reg.sized) {
            fail_malformed("'" + name.text + "' is not a register and cannot be indexed", open);
        }
        if (!index.is_int || index.value < 0 || index.value >= reg.size) {
            fail_malformed("index out of range for '" + name.text + "'", open);
        }
        return static_cast<std::uint32_t>(index.value);
    }

    void add_measures(const std::vector<std::uint32_t>& qubits,
                      const std::vector<std::uint32_t>& clbits,
                      const Token& where) {
        if (qubits.size() != clbits.size()) {
            fail_malformed("a measurement's qubit and bit counts differ", where);
        }
        for (std::size_t i = 0; i < qubits.size(); ++i) {
            operations_.emplace_back(MeasureOp{qubits[i], clbits[i]});
        }
    }

    // measure q -> c;
    void parse_measure_arrow() {
        const Token& keyword = advance();
        const std::vector<std::uint32_t> qubits = parse_bits(true);
        if (peek().kind == TokenKind::Semicolon) {
            fail_unsupported("'measure' without a target bit is not supported", keyword);
        }
        expect(TokenKind::Arrow, "'->' in a measure statement");
        const std::vector<std::uint32_t> clbits = parse_bits(false);
        expect(TokenKind::Semicolon, "';' after a measure statement");
        add_measures(qubits, clbits, keyword);
    }

    // c = measure q;  c[i] = measure q[j];
    void parse_measure_assignment() {
        const Token& start = peek();
        const std::vector<std::uint32_t> clbits = parse_bits(false);
        expect(TokenKind::Equals, "'='");
        if (!at_ident("measure")) {
            fail_unsupported("classical assignments are not supported (only '= measure')", start);
        }
        advance();
        const std::vector<std::uint32_t> qubits = parse_bits(true);
        expect(TokenKind::Semicolon, "';' after a measure statement");
        add_measures(qubits, clbits, start);
    }

    // ---- gate calls -------------------------------------------------------

    void emit(std::vector<Instr> steps) {
        for (Instr& step : steps) {
            if (const auto* phase = std::get_if<GlobalPhase>(&step)) {
                global_phase_ += phase->angle;
            } else if (auto* gate = std::get_if<GateOp>(&step)) {
                operations_.emplace_back(std::move(*gate));
            } else {
                operations_.emplace_back(std::move(std::get<UnitaryOp>(step)));
            }
        }
    }

    std::vector<Modifier> parse_modifiers() {
        std::vector<Modifier> modifiers;
        while (peek().kind == TokenKind::Ident && is_modifier(peek().text)) {
            const Token& keyword = advance();
            Modifier modifier{Modifier::Kind::Inv, 1};
            if (keyword.text == "pow") {
                modifier.kind = Modifier::Kind::Pow;
                expect(TokenKind::LParen, "'(' after 'pow'");
                const Value k = parse_expr();
                expect(TokenKind::RParen, "')' after the exponent");
                if (!std::isfinite(k.value) || std::trunc(k.value) != k.value ||
                    std::abs(k.value) > 1e9) {
                    fail_unsupported("'pow' with a non-integer exponent is not supported", keyword);
                }
                modifier.count = static_cast<std::int64_t>(k.value);
            } else if (keyword.text == "ctrl" || keyword.text == "negctrl") {
                modifier.kind =
                    keyword.text == "ctrl" ? Modifier::Kind::Ctrl : Modifier::Kind::NegCtrl;
                if (peek().kind == TokenKind::LParen) {
                    advance();
                    const Value n = parse_expr();
                    expect(TokenKind::RParen, "')' after the control count");
                    if (!n.is_int || n.value < 1 || n.value > 64) {
                        fail_malformed("a control count must be a positive integer", keyword);
                    }
                    modifier.count = static_cast<std::int64_t>(n.value);
                }
            }
            expect(TokenKind::At, "'@' after a gate modifier");
            modifiers.push_back(modifier);
        }
        return modifiers;
    }

    QubitId parse_operand(const Scope* scope) {
        const Token& name = peek();
        if (name.kind == TokenKind::Other && name.text == "$") {
            fail_unsupported("physical qubits ('$n') are not supported", name);
        }
        expect(TokenKind::Ident, "a qubit operand");
        if (scope != nullptr) {
            const auto it = scope->qubits->find(name.text);
            if (it == scope->qubits->end()) {
                fail_malformed("'" + name.text + "' is not a qubit argument of this gate", name);
            }
            if (peek().kind == TokenKind::LBracket) {
                fail_malformed("a gate's qubit arguments cannot be indexed", peek());
            }
            return it->second;
        }
        const auto it = qregs_.find(name.text);
        if (it == qregs_.end()) {
            fail_malformed("undeclared qubit register '" + name.text + "'", name);
        }
        if (peek().kind == TokenKind::LBracket) {
            return it->second.offset + parse_index(it->second, name);
        }
        if (it->second.sized) {
            fail_unsupported("applying a gate to a whole register is not supported", name);
        }
        return it->second.offset;
    }

    // [modifiers] name[(params)] operands;  inside a body when scope is set.
    std::vector<Instr> parse_gate_call(const Scope* scope) {
        const Token& start = peek();
        const std::vector<Modifier> modifiers = parse_modifiers();
        const Token& name = expect(TokenKind::Ident, "a gate name");
        std::vector<double> params;
        if (peek().kind == TokenKind::LParen) {
            advance();
            if (peek().kind != TokenKind::RParen) {
                params.push_back(parse_expr(scope).value);
                while (peek().kind == TokenKind::Comma) {
                    advance();
                    params.push_back(parse_expr(scope).value);
                }
            }
            expect(TokenKind::RParen, "')' after gate parameters");
        }
        std::vector<QubitId> operands;
        if (peek().kind != TokenKind::Semicolon) {
            operands.push_back(parse_operand(scope));
            while (peek().kind == TokenKind::Comma) {
                advance();
                operands.push_back(parse_operand(scope));
            }
        }
        expect(TokenKind::Semicolon, "';' after a gate call");

        std::size_t num_controls = 0;
        for (const Modifier& m : modifiers) {
            if (m.kind == Modifier::Kind::Ctrl || m.kind == Modifier::Kind::NegCtrl) {
                num_controls += static_cast<std::size_t>(m.count);
            }
        }
        if (operands.size() < num_controls) {
            fail_malformed("too few qubits for the control modifiers", start);
        }
        const std::vector<QubitId> targets(
            operands.begin() + static_cast<std::ptrdiff_t>(num_controls), operands.end());
        std::vector<Instr> body = expand_base(name, params, targets, scope);

        // Modifiers apply from the gate outwards; the controls of the
        // outermost modifier are the leftmost operands.
        std::size_t control_end = num_controls;
        const std::string where = location(start);
        for (auto it = modifiers.rbegin(); it != modifiers.rend(); ++it) {
            switch (it->kind) {
                case Modifier::Kind::Inv:
                    body = detail::invert(std::move(body));
                    break;
                case Modifier::Kind::Pow:
                    body = detail::power(body, it->count);
                    break;
                case Modifier::Kind::Ctrl:
                case Modifier::Kind::NegCtrl: {
                    const std::size_t count = static_cast<std::size_t>(it->count);
                    for (std::size_t i = 0; i < count; ++i) {
                        const QubitId c = operands[control_end - 1 - i];
                        body = it->kind == Modifier::Kind::Ctrl
                                   ? detail::control(body, c, where)
                                   : detail::negative_control(body, c, where);
                    }
                    control_end -= count;
                    break;
                }
            }
        }
        return body;
    }

    std::vector<Instr> expand_base(const Token& name,
                                   const std::vector<double>& params,
                                   const std::vector<QubitId>& targets,
                                   const Scope* scope) {
        const auto check = [&](std::size_t num_params, std::size_t num_qubits) {
            if (params.size() != num_params) {
                fail_malformed("gate '" + name.text + "' takes " + std::to_string(num_params) +
                                   " parameter(s), got " + std::to_string(params.size()),
                               name);
            }
            if (targets.size() != num_qubits) {
                fail_malformed("gate '" + name.text + "' takes " + std::to_string(num_qubits) +
                                   " qubit(s), got " + std::to_string(targets.size()),
                               name);
            }
        };
        if (name.text == "U") {
            check(3, 1);
            return {GateOp{Opcode::U, targets, params}};
        }
        if (name.text == "gphase") {
            check(1, 0);
            return {GlobalPhase{params[0]}};
        }
        if (const auto it = gates_.find(name.text); it != gates_.end()) {
            if (scope != nullptr && it->second.order >= scope->gate_order) {
                fail_malformed("gate '" + name.text + "' is used before its definition", name);
            }
            check(it->second.params.size(), it->second.qubits.size());
            if (it->second.canonical_rzz) {
                return {GateOp{Opcode::RZZ, targets, params}};
            }
            return expand_user_gate(it->second, params, targets);
        }
        if (stdgates_) {
            if (const detail::StdGate* gate = detail::find_stdgate(name.text)) {
                check(static_cast<std::size_t>(gate->num_params),
                      static_cast<std::size_t>(gate->num_qubits));
                return detail::expand_stdgate(*gate, params, targets);
            }
        }
        if (const auto it = refused_keywords().find(name.text); it != refused_keywords().end()) {
            fail_unsupported(std::string(it->second), name);
        }
        fail_malformed("undefined gate '" + name.text + "'", name);
    }

    std::vector<Instr> expand_user_gate(const GateDef& def,
                                        const std::vector<double>& params,
                                        const std::vector<QubitId>& targets) {
        NameMap<double> param_values;
        for (std::size_t i = 0; i < def.params.size(); ++i) {
            param_values.emplace(def.params[i], params[i]);
        }
        NameMap<QubitId> qubit_values;
        for (std::size_t i = 0; i < def.qubits.size(); ++i) {
            qubit_values.emplace(def.qubits[i], targets[i]);
        }
        const Scope scope{&param_values, &qubit_values, def.order};

        const std::size_t saved = pos_;
        pos_ = def.body_begin;
        std::vector<Instr> body;
        while (pos_ < def.body_end) {
            if (at_ident("barrier")) {
                skip_statement();
                continue;
            }
            if (peek().kind != TokenKind::Ident) {
                fail_malformed("expected a gate call in a gate body", peek());
            }
            if (const auto it = refused_keywords().find(peek().text);
                it != refused_keywords().end()) {
                fail_unsupported(std::string(it->second), peek());
            }
            if (peek().text == "measure") {
                fail_malformed("a gate body contains only gate calls", peek());
            }
            std::vector<Instr> step = parse_gate_call(&scope);
            body.insert(body.end(),
                        std::make_move_iterator(step.begin()),
                        std::make_move_iterator(step.end()));
        }
        pos_ = saved;
        return body;
    }

    // ---- constant expressions --------------------------------------------

    Value parse_expr(const Scope* scope = nullptr) {
        Value left = parse_term(scope);
        while (peek().kind == TokenKind::Plus || peek().kind == TokenKind::Minus) {
            const bool plus = advance().kind == TokenKind::Plus;
            const Value right = parse_term(scope);
            left = Value{plus ? left.value + right.value : left.value - right.value,
                         left.is_int && right.is_int};
        }
        return left;
    }

    Value parse_term(const Scope* scope) {
        Value left = parse_unary(scope);
        while (peek().kind == TokenKind::Star || peek().kind == TokenKind::Slash) {
            const Token& op = advance();
            const Value right = parse_unary(scope);
            if (op.kind == TokenKind::Star) {
                left = Value{left.value * right.value, left.is_int && right.is_int};
            } else {
                if (left.is_int && right.is_int) {
                    fail_unsupported(
                        "division of two integers is not supported; write a "
                        "real literal such as 1.0",
                        op);
                }
                left = Value{left.value / right.value, false};
            }
        }
        return left;
    }

    Value parse_unary(const Scope* scope) {
        if (peek().kind == TokenKind::Minus) {
            advance();
            const Value v = parse_unary(scope);
            return Value{-v.value, v.is_int};
        }
        if (peek().kind == TokenKind::Plus) {
            advance();
            return parse_unary(scope);
        }
        return parse_power(scope);
    }

    Value parse_power(const Scope* scope) {
        const Value base = parse_primary(scope);
        if (peek().kind == TokenKind::Power) {
            advance();
            const Value exponent = parse_unary(scope);
            return Value{std::pow(base.value, exponent.value),
                         base.is_int && exponent.is_int && exponent.value >= 0};
        }
        return base;
    }

    Value parse_primary(const Scope* scope) {
        const Token& token = peek();
        if (token.kind == TokenKind::Int || token.kind == TokenKind::Real) {
            advance();
            return Value{token.number, token.kind == TokenKind::Int};
        }
        if (token.kind == TokenKind::LParen) {
            advance();
            const Value v = parse_expr(scope);
            expect(TokenKind::RParen, "')'");
            return v;
        }
        if (token.kind != TokenKind::Ident) {
            fail_malformed("expected an expression", token);
        }
        advance();
        const std::string& name = token.text;
        if (name == "pi" || name == "π") {
            return Value{std::numbers::pi, false};
        }
        if (name == "tau" || name == "τ") {
            return Value{2.0 * std::numbers::pi, false};
        }
        if (name == "euler" || name == "ℯ") {
            return Value{std::numbers::e, false};
        }
        if (peek().kind == TokenKind::LParen) {
            return call_function(token, scope);
        }
        if (scope != nullptr) {
            if (const auto it = scope->params->find(name); it != scope->params->end()) {
                return Value{it->second, false};
            }
        }
        if (const auto it = constants_.find(name); it != constants_.end()) {
            return it->second;
        }
        if (name == "measure") {
            fail_unsupported("a measurement used as a value is not supported", token);
        }
        fail_malformed("'" + name + "' is not a constant, an input or a gate parameter", token);
    }

    Value call_function(const Token& name, const Scope* scope) {
        using Fn = double (*)(double);
        static const NameMap<Fn> kFunctions = {
            {"sin", [](double x) { return std::sin(x); }},
            {"cos", [](double x) { return std::cos(x); }},
            {"tan", [](double x) { return std::tan(x); }},
            {"arcsin", [](double x) { return std::asin(x); }},
            {"arccos", [](double x) { return std::acos(x); }},
            {"arctan", [](double x) { return std::atan(x); }},
            {"exp", [](double x) { return std::exp(x); }},
            {"ln", [](double x) { return std::log(x); }},
            {"sqrt", [](double x) { return std::sqrt(x); }},
        };
        const auto it = kFunctions.find(name.text);
        if (it == kFunctions.end()) {
            fail_unsupported("function '" + name.text + "' is not supported", name);
        }
        expect(TokenKind::LParen, "'('");
        const Value argument = parse_expr(scope);
        expect(TokenKind::RParen, "')' after a function argument");
        return Value{it->second(argument.value), false};
    }

    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
    const InputBindings& inputs_;
    std::set<std::string, std::less<>> used_inputs_;

    NameMap<Register> qregs_;
    NameMap<Register> cregs_;
    std::uint32_t num_qubits_ = 0;
    std::uint32_t num_clbits_ = 0;
    NameMap<Value> constants_;
    NameMap<GateDef> gates_;
    bool stdgates_ = false;

    std::vector<Operation> operations_;
    double global_phase_ = 0.0;
};

}  // namespace

Circuit parse(std::string_view text, const InputBindings& inputs) {
    Parser parser(detail::tokenize(text), inputs);
    return parser.run();
}

}  // namespace menos::qasm3
