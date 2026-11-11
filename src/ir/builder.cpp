#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include "common/error.h"
#include "ir/identifier.h"
#include "ir/inverse.h"
#include "menos/builder.h"

namespace menos {
namespace {

std::string join(const std::vector<std::string>& names) {
    std::string out;
    for (const std::string& name : names) {
        if (!out.empty()) {
            out += ", ";
        }
        out += "'" + name + "'";
    }
    return out;
}

const char* kind_of(const BuilderOperation& op) {
    if (std::holds_alternative<BuilderGate>(op)) {
        return "a gate";
    }
    if (std::holds_alternative<UnitaryOp>(op)) {
        return "a unitary";
    }
    if (std::holds_alternative<MeasureOp>(op)) {
        return "a measurement";
    }
    return "a channel";
}

}  // namespace

QubitId QuantumRegister::operator[](std::uint32_t i) const {
    if (i >= size) {
        throw common::InvalidCircuit("register '" + name + "' has " + std::to_string(size) +
                                     " qubits; index " + std::to_string(i) + " is out of range");
    }
    return offset + i;
}

ClbitId ClassicalRegister::operator[](std::uint32_t i) const {
    if (i >= size) {
        throw common::InvalidCircuit("register '" + name + "' has " + std::to_string(size) +
                                     " bits; index " + std::to_string(i) + " is out of range");
    }
    return offset + i;
}

CircuitBuilder::CircuitBuilder(std::uint32_t num_qubits, std::uint32_t num_clbits) {
    if (num_qubits > 0) {
        add_qreg("q", num_qubits);
    }
    if (num_clbits > 0) {
        add_creg("c", num_clbits);
    }
}

CircuitBuilder CircuitBuilder::from_circuit(const Circuit& circuit) {
    CircuitBuilder out(circuit.num_qubits(), circuit.num_clbits());
    out.global_phase = circuit.global_phase;
    out.operations_.reserve(circuit.operations().size());
    for (const Operation& op : circuit.operations()) {
        std::visit(
            [&out](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, GateOp>) {
                    std::vector<ParameterExpression> params(o.params.begin(), o.params.end());
                    out.operations_.emplace_back(BuilderGate{o.opcode, o.qubits, std::move(params)});
                } else {
                    out.operations_.emplace_back(o);
                }
            },
            op);
    }
    return out;
}

void CircuitBuilder::check_register_name(const std::string& name) const {
    if (!ir::is_identifier(name)) {
        throw common::InvalidCircuit("register name '" + name +
                                     "' is not an identifier ([A-Za-z_][A-Za-z0-9_]*)");
    }
    const bool taken =
        std::any_of(qregs_.begin(), qregs_.end(), [&](const auto& r) { return r.name == name; }) ||
        std::any_of(cregs_.begin(), cregs_.end(), [&](const auto& r) { return r.name == name; });
    if (taken) {
        throw common::InvalidCircuit("register name '" + name + "' is already declared");
    }
}

QuantumRegister CircuitBuilder::add_qreg(std::string name, std::uint32_t size) {
    check_register_name(name);
    if (size == 0) {
        throw common::InvalidCircuit("register '" + name + "' must have at least one qubit");
    }
    QuantumRegister reg{std::move(name), num_qubits_, size};
    num_qubits_ += size;
    qregs_.push_back(reg);
    return reg;
}

ClassicalRegister CircuitBuilder::add_creg(std::string name, std::uint32_t size) {
    check_register_name(name);
    if (size == 0) {
        throw common::InvalidCircuit("register '" + name + "' must have at least one bit");
    }
    ClassicalRegister reg{std::move(name), num_clbits_, size};
    num_clbits_ += size;
    cregs_.push_back(reg);
    return reg;
}

const QuantumRegister& CircuitBuilder::qreg(std::string_view name) const {
    for (const QuantumRegister& reg : qregs_) {
        if (reg.name == name) {
            return reg;
        }
    }
    throw common::InvalidCircuit("no quantum register named '" + std::string(name) + "'");
}

const ClassicalRegister& CircuitBuilder::creg(std::string_view name) const {
    for (const ClassicalRegister& reg : cregs_) {
        if (reg.name == name) {
            return reg;
        }
    }
    throw common::InvalidCircuit("no classical register named '" + std::string(name) + "'");
}

void CircuitBuilder::check_qubit(QubitId q) const {
    if (q >= num_qubits_) {
        throw common::InvalidCircuit("qubit " + std::to_string(q) + " is outside the circuit's " +
                                     std::to_string(num_qubits_) + " qubits");
    }
}

void CircuitBuilder::check_clbit(ClbitId c) const {
    if (c >= num_clbits_) {
        throw common::InvalidCircuit("classical bit " + std::to_string(c) +
                                     " is outside the circuit's " + std::to_string(num_clbits_) +
                                     " classical bits");
    }
}

CircuitBuilder& CircuitBuilder::gate(Opcode opcode,
                                     std::vector<QubitId> qubits,
                                     std::vector<ParameterExpression> params) {
    for (const QubitId q : qubits) {
        check_qubit(q);
    }
    operations_.emplace_back(BuilderGate{opcode, std::move(qubits), std::move(params)});
    return *this;
}

CircuitBuilder& CircuitBuilder::x(QubitId q) { return gate(Opcode::X, {q}); }
CircuitBuilder& CircuitBuilder::y(QubitId q) { return gate(Opcode::Y, {q}); }
CircuitBuilder& CircuitBuilder::z(QubitId q) { return gate(Opcode::Z, {q}); }
CircuitBuilder& CircuitBuilder::h(QubitId q) { return gate(Opcode::H, {q}); }
CircuitBuilder& CircuitBuilder::s(QubitId q) { return gate(Opcode::S, {q}); }
CircuitBuilder& CircuitBuilder::sdg(QubitId q) { return gate(Opcode::Sdg, {q}); }
CircuitBuilder& CircuitBuilder::t(QubitId q) { return gate(Opcode::T, {q}); }
CircuitBuilder& CircuitBuilder::tdg(QubitId q) { return gate(Opcode::Tdg, {q}); }
CircuitBuilder& CircuitBuilder::sx(QubitId q) { return gate(Opcode::SX, {q}); }
CircuitBuilder& CircuitBuilder::sxdg(QubitId q) { return gate(Opcode::SXdg, {q}); }
CircuitBuilder& CircuitBuilder::rx(const Angle& theta, QubitId q) {
    return gate(Opcode::RX, {q}, {theta});
}
CircuitBuilder& CircuitBuilder::ry(const Angle& theta, QubitId q) {
    return gate(Opcode::RY, {q}, {theta});
}
CircuitBuilder& CircuitBuilder::rz(const Angle& theta, QubitId q) {
    return gate(Opcode::RZ, {q}, {theta});
}
CircuitBuilder& CircuitBuilder::p(const Angle& lambda, QubitId q) {
    return gate(Opcode::P, {q}, {lambda});
}
CircuitBuilder& CircuitBuilder::u(const Angle& theta,
                                  const Angle& phi,
                                  const Angle& lambda,
                                  QubitId q) {
    return gate(Opcode::U, {q}, {theta, phi, lambda});
}
CircuitBuilder& CircuitBuilder::cx(QubitId control, QubitId target) {
    return gate(Opcode::CX, {control, target});
}
CircuitBuilder& CircuitBuilder::cz(QubitId control, QubitId target) {
    return gate(Opcode::CZ, {control, target});
}
CircuitBuilder& CircuitBuilder::swap(QubitId a, QubitId b) { return gate(Opcode::SWAP, {a, b}); }
CircuitBuilder& CircuitBuilder::cp(const Angle& lambda, QubitId control, QubitId target) {
    return gate(Opcode::CP, {control, target}, {lambda});
}
CircuitBuilder& CircuitBuilder::rzz(const Angle& theta, QubitId a, QubitId b) {
    return gate(Opcode::RZZ, {a, b}, {theta});
}
CircuitBuilder& CircuitBuilder::ccx(QubitId control1, QubitId control2, QubitId target) {
    return gate(Opcode::CCX, {control1, control2, target});
}

CircuitBuilder& CircuitBuilder::unitary(std::vector<std::complex<double>> matrix,
                                        std::vector<QubitId> qubits) {
    for (const QubitId q : qubits) {
        check_qubit(q);
    }
    operations_.emplace_back(UnitaryOp{std::move(qubits), std::move(matrix)});
    return *this;
}

CircuitBuilder& CircuitBuilder::measure(QubitId qubit,
                                        ClbitId clbit,
                                        std::optional<ReadoutError> readout_error) {
    check_qubit(qubit);
    check_clbit(clbit);
    operations_.emplace_back(MeasureOp{qubit, clbit, readout_error});
    return *this;
}

CircuitBuilder& CircuitBuilder::measure_all() {
    if (num_qubits_ == 0) {
        throw common::InvalidCircuit("measure_all: the circuit has no qubits");
    }
    const ClassicalRegister meas = add_creg("meas", num_qubits_);
    for (QubitId q = 0; q < num_qubits_; ++q) {
        operations_.emplace_back(MeasureOp{q, meas.offset + q, std::nullopt});
    }
    return *this;
}

CircuitBuilder& CircuitBuilder::channel(std::vector<std::vector<std::complex<double>>> kraus,
                                        std::vector<QubitId> qubits) {
    for (const QubitId q : qubits) {
        check_qubit(q);
    }
    operations_.emplace_back(ChannelOp{std::move(qubits), std::move(kraus)});
    return *this;
}

namespace {

// The circuit-wide index map of compose(): identity when map is empty.
std::vector<std::uint32_t> resolve_map(const std::vector<std::uint32_t>& map,
                                       std::uint32_t other_size,
                                       std::uint32_t own_size,
                                       const char* what) {
    std::vector<std::uint32_t> out;
    if (map.empty()) {
        if (other_size > own_size) {
            throw common::InvalidCircuit(std::string("compose: the other circuit has ") +
                                         std::to_string(other_size) + " " + what +
                                         ", more than this circuit's " +
                                         std::to_string(own_size));
        }
        out.resize(other_size);
        for (std::uint32_t i = 0; i < other_size; ++i) {
            out[i] = i;
        }
        return out;
    }
    if (map.size() != other_size) {
        throw common::InvalidCircuit(std::string("compose: the ") + what + " map has " +
                                     std::to_string(map.size()) +
                                     " entries; the other circuit has " +
                                     std::to_string(other_size) + " " + what);
    }
    std::set<std::uint32_t> seen;
    for (const std::uint32_t index : map) {
        if (index >= own_size) {
            throw common::InvalidCircuit(std::string("compose: ") + what + " index " +
                                         std::to_string(index) + " is outside this circuit's " +
                                         std::to_string(own_size) + " " + what);
        }
        if (!seen.insert(index).second) {
            throw common::InvalidCircuit(std::string("compose: ") + what + " index " +
                                         std::to_string(index) + " appears twice in the map");
        }
    }
    return map;
}

std::vector<QubitId> remap(const std::vector<QubitId>& qubits,
                           const std::vector<std::uint32_t>& map) {
    std::vector<QubitId> out;
    out.reserve(qubits.size());
    for (const QubitId q : qubits) {
        // An operand outside other's qubits is left for ir::validate to
        // report at run (the builder never wrote one, but a from_circuit
        // builder may hold one).
        out.push_back(q < map.size() ? map[q] : q);
    }
    return out;
}

}  // namespace

CircuitBuilder& CircuitBuilder::compose(const CircuitBuilder& other,
                                        const std::vector<QubitId>& qubits,
                                        const std::vector<ClbitId>& clbits) {
    const std::vector<std::uint32_t> qmap =
        resolve_map(qubits, other.num_qubits_, num_qubits_, "qubits");
    const std::vector<std::uint32_t> cmap =
        resolve_map(clbits, other.num_clbits_, num_clbits_, "classical bits");
    std::vector<BuilderOperation> appended;
    appended.reserve(other.operations_.size());
    for (const BuilderOperation& op : other.operations_) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                T copy = o;
                if constexpr (std::is_same_v<T, MeasureOp>) {
                    copy.qubit = o.qubit < qmap.size() ? qmap[o.qubit] : o.qubit;
                    copy.clbit = o.clbit < cmap.size() ? cmap[o.clbit] : o.clbit;
                } else {
                    copy.qubits = remap(o.qubits, qmap);
                }
                appended.emplace_back(std::move(copy));
            },
            op);
    }
    operations_.insert(operations_.end(),
                       std::make_move_iterator(appended.begin()),
                       std::make_move_iterator(appended.end()));
    global_phase = global_phase + other.global_phase;
    return *this;
}

CircuitBuilder CircuitBuilder::inverse() const {
    CircuitBuilder out = *this;
    out.operations_.clear();
    out.operations_.reserve(operations_.size());
    for (std::size_t i = operations_.size(); i-- > 0;) {
        const BuilderOperation& op = operations_[i];
        if (const auto* g = std::get_if<BuilderGate>(&op)) {
            auto [opcode, params] = ir::inverse_gate(g->opcode, g->params);
            out.operations_.emplace_back(BuilderGate{opcode, g->qubits, std::move(params)});
        } else if (const auto* u = std::get_if<UnitaryOp>(&op)) {
            out.operations_.emplace_back(ir::inverse_of(*u));
        } else {
            throw common::InvalidCircuit("inverse: operation " + std::to_string(i) + " is " +
                                         kind_of(op) + ", which has no inverse");
        }
    }
    // 0.0 - x is exactly -x for every x except a zero phase, which stays +0.0
    // (so a phase-free circuit's inverse is phase-free, and exports as such).
    out.global_phase = 0.0 - global_phase;
    return out;
}

CircuitBuilder CircuitBuilder::repeat(std::uint32_t k) const {
    CircuitBuilder out = *this;
    out.operations_.clear();
    out.operations_.reserve(operations_.size() * k);
    out.global_phase = 0.0;
    for (std::uint32_t copy = 0; copy < k; ++copy) {
        out.operations_.insert(out.operations_.end(), operations_.begin(), operations_.end());
        out.global_phase = out.global_phase + global_phase;
    }
    return out;
}

std::vector<std::string> CircuitBuilder::parameters() const {
    std::set<std::string> names;
    for (const BuilderOperation& op : operations_) {
        if (const auto* g = std::get_if<BuilderGate>(&op)) {
            for (const ParameterExpression& e : g->params) {
                for (const ParameterExpression::Term& term : e.terms()) {
                    names.insert(term.name);
                }
            }
        }
    }
    return {names.begin(), names.end()};
}

Circuit CircuitBuilder::bind(const std::map<std::string, double, std::less<>>& values) const {
    const std::vector<std::string> names = parameters();
    std::vector<std::string> missing;
    for (const std::string& name : names) {
        if (!values.contains(name)) {
            missing.push_back(name);
        }
    }
    if (!missing.empty()) {
        throw common::InvalidCircuit("unbound parameters: " + join(missing));
    }
    std::vector<std::string> unknown;
    std::vector<std::string> non_finite;
    for (const auto& [name, value] : values) {
        if (!std::binary_search(names.begin(), names.end(), name)) {
            unknown.push_back(name);
        } else if (!std::isfinite(value)) {
            non_finite.push_back(name);
        }
    }
    if (!unknown.empty()) {
        throw common::InvalidCircuit("bindings name no parameter of the circuit: " + join(unknown));
    }
    if (!non_finite.empty()) {
        throw common::InvalidCircuit("parameters bound to a non-finite value: " +
                                     join(non_finite));
    }

    Circuit circuit(num_qubits_, num_clbits_);
    circuit.global_phase = global_phase;
    for (const BuilderOperation& op : operations_) {
        std::visit(
            [&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, BuilderGate>) {
                    std::vector<double> params;
                    params.reserve(o.params.size());
                    for (const ParameterExpression& e : o.params) {
                        params.push_back(e.evaluate(values));
                    }
                    circuit.append(o.opcode, o.qubits, std::move(params));
                } else if constexpr (std::is_same_v<T, UnitaryOp>) {
                    circuit.append_unitary(o.qubits, o.matrix);
                } else if constexpr (std::is_same_v<T, MeasureOp>) {
                    circuit.append_measure(o.qubit, o.clbit, o.readout_error);
                } else {
                    circuit.append_channel(o.qubits, o.kraus);
                }
            },
            op);
    }
    return circuit;
}

Circuit CircuitBuilder::to_circuit() const { return bind({}); }

}  // namespace menos
