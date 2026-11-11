#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "menos/circuit.h"

namespace menos {

// The circuit builder (docs/adr/0029, Phase 25): named gate methods, run-time
// parameters, composition, inverse, repeat, named registers and OpenQASM 3
// export (menos::qasm3::emit). It adds no operation kind: every circuit it
// produces is an ordinary Circuit, built through Circuit::append*, and is
// planned by plan_job exactly as one built by hand.
//
// Every exception below is menos::common::InvalidCircuit unless stated.

// A named run-time parameter. Its identity is its name (two Parameter("theta")
// are the same parameter, also across CircuitBuilder::compose). The name must
// match [A-Za-z_][A-Za-z0-9_]*, so it can be exported as an OpenQASM 3
// `input`; any other name throws.
class Parameter {
public:
    explicit Parameter(std::string name);
    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

private:
    std::string name_;
};

// An affine expression c1*t1 + c2*t2 + ... (+ c0) of parameters, or a plain
// constant (no terms). Terms keep the order in which their parameters first
// appear. Built with + - * (and unary -) from parameters, expressions and
// doubles; multiplying two expressions that both have parameters throws.
//
// Evaluation order, fixed so bound circuits are reproducible bit for bit
// (docs/adr/0029 section 3.2): the products c_i*v_i are summed left to right,
// ((c1*v1 + c2*v2) + ...), and c0, when the expression has a constant, is
// added last. A bare parameter (coefficient 1) evaluates to exactly its value
// and its negation to exactly minus its value; a constant evaluates to
// itself.
class ParameterExpression {
public:
    struct Term {
        std::string name;
        double coefficient;
    };

    // A constant. Implicit, so a double is accepted wherever an angle is.
    ParameterExpression(double constant);  // NOLINT(google-explicit-constructor)
    // The parameter with coefficient 1. Implicit for the same reason.
    ParameterExpression(const Parameter& parameter);  // NOLINT(google-explicit-constructor)

    [[nodiscard]] const std::vector<Term>& terms() const noexcept {
        return terms_;
    }
    // c0, or nullopt when the expression has none (a bare parameter).
    [[nodiscard]] const std::optional<double>& constant() const noexcept {
        return constant_;
    }
    [[nodiscard]] bool is_constant() const noexcept {
        return terms_.empty();
    }

    // The value under values (by parameter name), in the order above. Every
    // term's parameter must have a value: InternalError otherwise (callers
    // check bindings first, see CircuitBuilder::bind).
    [[nodiscard]] double evaluate(
        const std::map<std::string, double, std::less<>>& values = {}) const;

    friend ParameterExpression operator-(const ParameterExpression& a);
    friend ParameterExpression operator+(const ParameterExpression& a,
                                         const ParameterExpression& b);
    friend ParameterExpression operator*(const ParameterExpression& a,
                                         const ParameterExpression& b);

private:
    ParameterExpression() = default;
    std::vector<Term> terms_;
    std::optional<double> constant_;
};

// -a: every coefficient and the constant negated (exact).
ParameterExpression operator-(const ParameterExpression& a);
// a + b: terms of a, then b's new parameters; a parameter in both gets the sum
// of its coefficients; the constants add (either alone is kept as is).
ParameterExpression operator+(const ParameterExpression& a, const ParameterExpression& b);
// a + (-b).
ParameterExpression operator-(const ParameterExpression& a, const ParameterExpression& b);
// A constant times an expression scales every coefficient and the constant.
// Throws if both a and b have parameters (not affine).
ParameterExpression operator*(const ParameterExpression& a, const ParameterExpression& b);

// A gate angle: a double converts to a constant expression.
using Angle = ParameterExpression;

// A contiguous range of qubit (or classical-bit) indices with a name.
// Registers are numbered in declaration order: the first register declared
// holds indices 0..size-1, the next continues after it (the OpenQASM 3
// importer's and Qiskit's rule, docs/adr/0029 section 3.5).
struct QuantumRegister {
    std::string name;
    std::uint32_t offset;
    std::uint32_t size;
    // The circuit-wide index of element i; throws if i >= size.
    [[nodiscard]] QubitId operator[](std::uint32_t i) const;
};

struct ClassicalRegister {
    std::string name;
    std::uint32_t offset;
    std::uint32_t size;
    [[nodiscard]] ClbitId operator[](std::uint32_t i) const;
};

// A named gate whose angles may be parameter expressions. Otherwise the same
// as GateOp.
struct BuilderGate {
    Opcode opcode;
    std::vector<QubitId> qubits;
    std::vector<ParameterExpression> params;
};

using BuilderOperation = std::variant<BuilderGate, UnitaryOp, MeasureOp, ChannelOp>;

class CircuitBuilder {
public:
    // Declares registers q[num_qubits] and c[num_clbits], each only when its
    // size is non-zero.
    explicit CircuitBuilder(std::uint32_t num_qubits = 0, std::uint32_t num_clbits = 0);

    // A builder holding circuit's operations, global phase and registers
    // q[num_qubits] and c[num_clbits] (no parameters).
    [[nodiscard]] static CircuitBuilder from_circuit(const Circuit& circuit);

    // Appends a register after the existing ones. The name must match
    // [A-Za-z_][A-Za-z0-9_]* and differ from every register's (quantum and
    // classical); the size must be at least 1.
    QuantumRegister add_qreg(std::string name, std::uint32_t size);
    ClassicalRegister add_creg(std::string name, std::uint32_t size);
    // The register of that name; throws if there is none.
    [[nodiscard]] const QuantumRegister& qreg(std::string_view name) const;
    [[nodiscard]] const ClassicalRegister& creg(std::string_view name) const;
    [[nodiscard]] const std::vector<QuantumRegister>& qregs() const noexcept {
        return qregs_;
    }
    [[nodiscard]] const std::vector<ClassicalRegister>& cregs() const noexcept {
        return cregs_;
    }

    [[nodiscard]] std::uint32_t num_qubits() const noexcept {
        return num_qubits_;
    }
    [[nodiscard]] std::uint32_t num_clbits() const noexcept {
        return num_clbits_;
    }

    // Named gates. Each appends exactly the operation Circuit::append(opcode,
    // qubits, params) would, with angles before qubits as in Qiskit, and
    // returns *this. A qubit index outside the declared registers throws at
    // the call; every other check is menos::ir::validate's, at run, as for
    // Circuit.
    CircuitBuilder& x(QubitId q);
    CircuitBuilder& y(QubitId q);
    CircuitBuilder& z(QubitId q);
    CircuitBuilder& h(QubitId q);
    CircuitBuilder& s(QubitId q);
    CircuitBuilder& sdg(QubitId q);
    CircuitBuilder& t(QubitId q);
    CircuitBuilder& tdg(QubitId q);
    CircuitBuilder& sx(QubitId q);
    CircuitBuilder& sxdg(QubitId q);
    CircuitBuilder& rx(const Angle& theta, QubitId q);
    CircuitBuilder& ry(const Angle& theta, QubitId q);
    CircuitBuilder& rz(const Angle& theta, QubitId q);
    CircuitBuilder& p(const Angle& lambda, QubitId q);
    CircuitBuilder& u(const Angle& theta, const Angle& phi, const Angle& lambda, QubitId q);
    CircuitBuilder& cx(QubitId control, QubitId target);
    CircuitBuilder& cz(QubitId control, QubitId target);
    CircuitBuilder& swap(QubitId a, QubitId b);
    CircuitBuilder& cp(const Angle& lambda, QubitId control, QubitId target);
    CircuitBuilder& rzz(const Angle& theta, QubitId a, QubitId b);
    CircuitBuilder& ccx(QubitId control1, QubitId control2, QubitId target);

    // Circuit::append_unitary(qubits, matrix).
    CircuitBuilder& unitary(std::vector<std::complex<double>> matrix, std::vector<QubitId> qubits);
    // Circuit::append_measure(qubit, clbit, readout_error); a clbit outside
    // the declared classical registers throws.
    CircuitBuilder& measure(QubitId qubit,
                            ClbitId clbit,
                            std::optional<ReadoutError> readout_error = std::nullopt);
    // Adds a classical register "meas" of num_qubits() bits and measures
    // qubit i into meas[i], as Qiskit's measure_all(). Throws if "meas" is
    // already a register name or the circuit has no qubits.
    CircuitBuilder& measure_all();
    // Circuit::append_channel(qubits, kraus).
    CircuitBuilder& channel(std::vector<std::vector<std::complex<double>>> kraus,
                            std::vector<QubitId> qubits);

    // Appends other's operations with other's qubit j sent to qubits[j] and
    // its classical bit j to clbits[j], and adds other's global phase. An
    // empty map is the identity (other must then be no wider than *this). A
    // non-empty map must have one entry per qubit (classical bit) of other,
    // distinct and in range. Parameters merge by name. Returns *this (unlike
    // Qiskit's compose, which returns a new circuit by default).
    CircuitBuilder& compose(const CircuitBuilder& other,
                            const std::vector<QubitId>& qubits = {},
                            const std::vector<ClbitId>& clbits = {});

    // The exact inverse: the operations in reverse order, each inverted by
    // menos::ir::inverse_gate (angles negated, U(t, p, l) -> U(-t, -l, -p),
    // s <-> sdg, ...), a unitary by its conjugate transpose, and the global
    // phase negated (computed as 0.0 - phase, so a zero phase stays +0.0).
    // Same registers. Throws, naming the operation's index, if
    // the circuit contains a measurement or a channel.
    [[nodiscard]] CircuitBuilder inverse() const;
    // k copies of the operations in order (k = 0 gives none), the global
    // phase added k times. Same registers.
    [[nodiscard]] CircuitBuilder repeat(std::uint32_t k) const;

    // Every parameter name used, sorted.
    [[nodiscard]] std::vector<std::string> parameters() const;
    [[nodiscard]] const std::vector<BuilderOperation>& operations() const noexcept {
        return operations_;
    }

    // An ordinary Circuit with every angle evaluated under values, built
    // through Circuit::append, append_unitary, append_measure and
    // append_channel. Throws, naming them, if a parameter has no value, a
    // value names no parameter of the circuit, or a value is not finite.
    [[nodiscard]] Circuit bind(const std::map<std::string, double, std::less<>>& values) const;
    // bind({}): throws, naming them, if the circuit has parameters.
    [[nodiscard]] Circuit to_circuit() const;

    // exp(i * global_phase) multiplies the state, as Circuit::global_phase.
    double global_phase = 0.0;

private:
    void check_qubit(QubitId q) const;
    void check_clbit(ClbitId c) const;
    void check_register_name(const std::string& name) const;
    CircuitBuilder& gate(Opcode opcode,
                         std::vector<QubitId> qubits,
                         std::vector<ParameterExpression> params = {});

    std::uint32_t num_qubits_ = 0;
    std::uint32_t num_clbits_ = 0;
    std::vector<QuantumRegister> qregs_;
    std::vector<ClassicalRegister> cregs_;
    std::vector<BuilderOperation> operations_;
};

}  // namespace menos
