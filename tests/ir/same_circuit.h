#pragma once

// Bitwise, operation-for-operation equality of two circuits, for tests of the
// circuit builder and the OpenQASM 3 exporter (docs/adr/0029).

#include <gtest/gtest.h>

#include <bit>
#include <complex>
#include <cstdint>
#include <variant>
#include <vector>

#include "menos/circuit.h"

namespace menos::test {

using Complex = std::complex<double>;

inline bool same_double(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

inline bool same_doubles(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_double(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

inline bool same_matrix(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_double(a[i].real(), b[i].real()) || !same_double(a[i].imag(), b[i].imag())) {
            return false;
        }
    }
    return true;
}

// Operation-for-operation, bitwise equality of two circuits.
inline ::testing::AssertionResult same_circuit(const Circuit& a, const Circuit& b) {
    if (a.num_qubits() != b.num_qubits() || a.num_clbits() != b.num_clbits()) {
        return ::testing::AssertionFailure() << "different widths";
    }
    if (!same_double(a.global_phase, b.global_phase)) {
        return ::testing::AssertionFailure() << "different global phase";
    }
    if (a.operations().size() != b.operations().size()) {
        return ::testing::AssertionFailure() << "different operation counts";
    }
    for (std::size_t i = 0; i < a.operations().size(); ++i) {
        const Operation& x = a.operations()[i];
        const Operation& y = b.operations()[i];
        if (x.index() != y.index()) {
            return ::testing::AssertionFailure() << "operation " << i << ": different kinds";
        }
        bool equal = false;
        if (const auto* g = std::get_if<GateOp>(&x)) {
            const auto& h = std::get<GateOp>(y);
            equal =
                g->opcode == h.opcode && g->qubits == h.qubits && same_doubles(g->params, h.params);
        } else if (const auto* u = std::get_if<UnitaryOp>(&x)) {
            const auto& v = std::get<UnitaryOp>(y);
            equal = u->qubits == v.qubits && same_matrix(u->matrix, v.matrix);
        } else if (const auto* m = std::get_if<MeasureOp>(&x)) {
            const auto& n = std::get<MeasureOp>(y);
            equal = m->qubit == n.qubit && m->clbit == n.clbit &&
                    m->readout_error.has_value() == n.readout_error.has_value() &&
                    (!m->readout_error.has_value() ||
                     m->readout_error->assignment == n.readout_error->assignment);
        } else {
            const auto& c = std::get<ChannelOp>(x);
            const auto& d = std::get<ChannelOp>(y);
            equal = c.qubits == d.qubits && c.kraus.size() == d.kraus.size();
            for (std::size_t k = 0; equal && k < c.kraus.size(); ++k) {
                equal = same_matrix(c.kraus[k], d.kraus[k]);
            }
        }
        if (!equal) {
            return ::testing::AssertionFailure() << "operation " << i << " differs";
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace menos::test
