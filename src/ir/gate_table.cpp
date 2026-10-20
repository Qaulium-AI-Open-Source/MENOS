#include "ir/gate_table.h"

#include <array>
#include <cmath>
#include <numbers>

#include "common/error.h"

namespace menos::ir {
namespace {

using Matrix = std::vector<std::complex<double>>;
using namespace std::complex_literals;

constexpr double kPi = std::numbers::pi_v<double>;

// --- 1-qubit, 0-parameter gates ---

Matrix matrix_x(std::span<const double>) {
    return {0.0, 1.0, 1.0, 0.0};
}

Matrix matrix_y(std::span<const double>) {
    return {0.0, -1.0i, 1.0i, 0.0};
}

Matrix matrix_z(std::span<const double>) {
    return {1.0, 0.0, 0.0, -1.0};
}

Matrix matrix_h(std::span<const double>) {
    const double s = 1.0 / std::sqrt(2.0);
    return {s, s, s, -s};
}

Matrix matrix_s(std::span<const double>) {
    return {1.0, 0.0, 0.0, 1.0i};
}

Matrix matrix_sdg(std::span<const double>) {
    return {1.0, 0.0, 0.0, -1.0i};
}

Matrix matrix_t(std::span<const double>) {
    return {1.0, 0.0, 0.0, std::exp(1.0i * (kPi / 4.0))};
}

Matrix matrix_tdg(std::span<const double>) {
    return {1.0, 0.0, 0.0, std::exp(-1.0i * (kPi / 4.0))};
}

Matrix matrix_sx(std::span<const double>) {
    return {0.5 * (1.0 + 1.0i), 0.5 * (1.0 - 1.0i), 0.5 * (1.0 - 1.0i), 0.5 * (1.0 + 1.0i)};
}

Matrix matrix_sxdg(std::span<const double>) {
    return {0.5 * (1.0 - 1.0i), 0.5 * (1.0 + 1.0i), 0.5 * (1.0 + 1.0i), 0.5 * (1.0 - 1.0i)};
}

// --- 1-qubit, parametrized gates ---

Matrix matrix_rx(std::span<const double> params) {
    const double theta = params[0];
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {c, -1.0i * s, -1.0i * s, c};
}

Matrix matrix_ry(std::span<const double> params) {
    const double theta = params[0];
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {c, -s, s, c};
}

Matrix matrix_rz(std::span<const double> params) {
    const double theta = params[0];
    return {std::exp(-1.0i * (theta / 2.0)), 0.0, 0.0, std::exp(1.0i * (theta / 2.0))};
}

Matrix matrix_p(std::span<const double> params) {
    const double lambda = params[0];
    return {1.0, 0.0, 0.0, std::exp(1.0i * lambda)};
}

Matrix matrix_u(std::span<const double> params) {
    const double theta = params[0];
    const double phi = params[1];
    const double lambda = params[2];
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {
        c,
        -std::exp(1.0i * lambda) * s,
        std::exp(1.0i * phi) * s,
        std::exp(1.0i * (phi + lambda)) * c,
    };
}

// --- 2-qubit gates. Operand order (a0, a1): bit 0 of the local index is
// a0, bit 1 is a1 (CLAUDE.md's operand-order rule). ---

Matrix matrix_cx(std::span<const double>) {
    // operands (control, target): control = bit 0, target = bit 1.
    return {
        1.0,
        0.0,
        0.0,
        0.0,  //
        0.0,
        0.0,
        0.0,
        1.0,  //
        0.0,
        0.0,
        1.0,
        0.0,  //
        0.0,
        1.0,
        0.0,
        0.0,
    };
}

Matrix matrix_cz(std::span<const double>) {
    return {
        1.0,
        0.0,
        0.0,
        0.0,  //
        0.0,
        1.0,
        0.0,
        0.0,  //
        0.0,
        0.0,
        1.0,
        0.0,  //
        0.0,
        0.0,
        0.0,
        -1.0,
    };
}

Matrix matrix_swap(std::span<const double>) {
    return {
        1.0,
        0.0,
        0.0,
        0.0,  //
        0.0,
        0.0,
        1.0,
        0.0,  //
        0.0,
        1.0,
        0.0,
        0.0,  //
        0.0,
        0.0,
        0.0,
        1.0,
    };
}

Matrix matrix_cp(std::span<const double> params) {
    const double lambda = params[0];
    Matrix m(16, 0.0);
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = std::exp(1.0i * lambda);
    return m;
}

// RZZ(theta) = exp(-i theta/2 Z(x)Z): diag(e^{-i t/2}, e^{i t/2}, e^{i t/2}, e^{-i t/2})
// in the operand-order bit convention (both operands symmetric). Matches
// Qiskit's RZZGate and PennyLane's IsingZZ, and equals CX RZ(theta) CX.
Matrix matrix_rzz(std::span<const double> params) {
    const double theta = params[0];
    const std::complex<double> same = std::exp(-0.5i * theta);  // |00>, |11>
    const std::complex<double> diff = std::exp(0.5i * theta);   // |01>, |10>
    Matrix m(16, 0.0);
    m[0] = same;
    m[5] = diff;
    m[10] = diff;
    m[15] = same;
    return m;
}

// --- 3-qubit gate. operands (control1, control2, target): control1 = bit
// 0, control2 = bit 1, target = bit 2. Flips the target (bit 2) when both
// controls are 1, i.e. swaps local indices 3 (0b011) and 7 (0b111). ---

Matrix matrix_ccx(std::span<const double>) {
    Matrix m(64, 0.0);
    for (std::size_t i = 0; i < 8; ++i) {
        m[i * 8 + i] = 1.0;
    }
    m[3 * 8 + 3] = 0.0;
    m[7 * 8 + 7] = 0.0;
    m[3 * 8 + 7] = 1.0;
    m[7 * 8 + 3] = 1.0;
    return m;
}

constexpr std::array<GateInfo, 20> kGateTable{{
    {Opcode::X, "x", 1, 0, &matrix_x, /*is_clifford=*/true},
    {Opcode::Y, "y", 1, 0, &matrix_y, /*is_clifford=*/true},
    {Opcode::Z, "z", 1, 0, &matrix_z, /*is_clifford=*/true},
    {Opcode::H, "h", 1, 0, &matrix_h, /*is_clifford=*/true},
    {Opcode::S, "s", 1, 0, &matrix_s, /*is_clifford=*/true},
    {Opcode::Sdg, "sdg", 1, 0, &matrix_sdg, /*is_clifford=*/true},
    {Opcode::T, "t", 1, 0, &matrix_t, /*is_clifford=*/false},
    {Opcode::Tdg, "tdg", 1, 0, &matrix_tdg, /*is_clifford=*/false},
    {Opcode::SX, "sx", 1, 0, &matrix_sx, /*is_clifford=*/true},
    {Opcode::SXdg, "sxdg", 1, 0, &matrix_sxdg, /*is_clifford=*/true},
    {Opcode::RX, "rx", 1, 1, &matrix_rx, /*is_clifford=*/false},
    {Opcode::RY, "ry", 1, 1, &matrix_ry, /*is_clifford=*/false},
    {Opcode::RZ, "rz", 1, 1, &matrix_rz, /*is_clifford=*/false},
    {Opcode::P, "p", 1, 1, &matrix_p, /*is_clifford=*/false},
    {Opcode::U, "u", 1, 3, &matrix_u, /*is_clifford=*/false},
    {Opcode::CX, "cx", 2, 0, &matrix_cx, /*is_clifford=*/true},
    {Opcode::CZ, "cz", 2, 0, &matrix_cz, /*is_clifford=*/true},
    {Opcode::SWAP, "swap", 2, 0, &matrix_swap, /*is_clifford=*/true},
    {Opcode::CP, "cp", 2, 1, &matrix_cp, /*is_clifford=*/false},
    {Opcode::RZZ, "rzz", 2, 1, &matrix_rzz, /*is_clifford=*/false},
}};

constexpr GateInfo kCcxInfo{Opcode::CCX, "ccx", 3, 0, &matrix_ccx, /*is_clifford=*/false};

}  // namespace

const GateInfo& gate_info(Opcode opcode) {
    if (opcode == Opcode::CCX) {
        return kCcxInfo;
    }
    for (const GateInfo& info : kGateTable) {
        if (info.opcode == opcode) {
            return info;
        }
    }
    throw common::InternalError(
        "gate_info: opcode has no gate-table entry (Unitary and Measure are not named gates)");
}

const GateInfo* gate_info_by_name(std::string_view name) {
    if (name == kCcxInfo.name) {
        return &kCcxInfo;
    }
    for (const GateInfo& info : kGateTable) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

std::vector<std::complex<double>> generate_matrix(Opcode opcode, std::span<const double> params) {
    const GateInfo& info = gate_info(opcode);
    common::ensure(static_cast<int>(params.size()) == info.num_params,
                   "generate_matrix: parameter count mismatch for gate " + std::string(info.name));
    return info.generator(params);
}

}  // namespace menos::ir
