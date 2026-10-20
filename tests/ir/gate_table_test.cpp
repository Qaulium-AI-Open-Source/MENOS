#include "ir/gate_table.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <string_view>
#include <vector>

#include "common/error.h"
#include "common/tolerance.h"

namespace menos::ir {
namespace {

// U*U^dagger == I within tolerance, for a dim x dim row-major matrix.
bool is_unitary(const std::vector<std::complex<double>>& m, std::size_t dim) {
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            std::complex<double> sum = 0.0;
            for (std::size_t k = 0; k < dim; ++k) {
                // (U U^dagger)[r][c] = sum_k U[r][k] * conj(U[c][k])
                sum += m[r * dim + k] * std::conj(m[c * dim + k]);
            }
            const std::complex<double> expected =
                (r == c) ? std::complex<double>{1.0, 0.0} : std::complex<double>{0.0, 0.0};
            if (std::abs(sum - expected) > common::kDefaultAbsTol) {
                return false;
            }
        }
    }
    return true;
}

struct GateTableParam {
    Opcode opcode;
    int dim;
    std::vector<double> params;
};

class GateUnitarityTest : public ::testing::TestWithParam<GateTableParam> {};

TEST_P(GateUnitarityTest, MatrixIsUnitary) {
    const GateTableParam& p = GetParam();
    const std::vector<std::complex<double>> matrix = generate_matrix(p.opcode, p.params);
    EXPECT_EQ(matrix.size(), static_cast<std::size_t>(p.dim) * static_cast<std::size_t>(p.dim));
    EXPECT_TRUE(is_unitary(matrix, static_cast<std::size_t>(p.dim)));
}

INSTANTIATE_TEST_SUITE_P(ZeroParamGates,
                         GateUnitarityTest,
                         ::testing::Values(GateTableParam{Opcode::X, 2, {}},
                                           GateTableParam{Opcode::Y, 2, {}},
                                           GateTableParam{Opcode::Z, 2, {}},
                                           GateTableParam{Opcode::H, 2, {}},
                                           GateTableParam{Opcode::S, 2, {}},
                                           GateTableParam{Opcode::Sdg, 2, {}},
                                           GateTableParam{Opcode::T, 2, {}},
                                           GateTableParam{Opcode::Tdg, 2, {}},
                                           GateTableParam{Opcode::SX, 2, {}},
                                           GateTableParam{Opcode::SXdg, 2, {}},
                                           GateTableParam{Opcode::CX, 4, {}},
                                           GateTableParam{Opcode::CZ, 4, {}},
                                           GateTableParam{Opcode::SWAP, 4, {}},
                                           GateTableParam{Opcode::CCX, 8, {}}));

INSTANTIATE_TEST_SUITE_P(ParametrizedGatesAtSeveralAngles,
                         GateUnitarityTest,
                         ::testing::Values(GateTableParam{Opcode::RX, 2, {0.0}},
                                           GateTableParam{Opcode::RX, 2, {0.7}},
                                           GateTableParam{Opcode::RX, 2, {-2.3}},
                                           GateTableParam{Opcode::RY, 2, {0.0}},
                                           GateTableParam{Opcode::RY, 2, {1.1}},
                                           GateTableParam{Opcode::RY, 2, {-3.14159}},
                                           GateTableParam{Opcode::RZ, 2, {0.0}},
                                           GateTableParam{Opcode::RZ, 2, {2.2}},
                                           GateTableParam{Opcode::P, 2, {0.0}},
                                           GateTableParam{Opcode::P, 2, {1.9}},
                                           GateTableParam{Opcode::CP, 4, {0.0}},
                                           GateTableParam{Opcode::CP, 4, {1.9}},
                                           GateTableParam{Opcode::RZZ, 4, {0.0}},
                                           GateTableParam{Opcode::RZZ, 4, {1.9}},
                                           GateTableParam{Opcode::RZZ, 4, {-2.6}},
                                           GateTableParam{Opcode::U, 2, {0.0, 0.0, 0.0}},
                                           GateTableParam{Opcode::U, 2, {0.7, 1.3, -0.4}},
                                           GateTableParam{Opcode::U, 2, {3.0, -1.5, 2.8}}));

TEST(GateTableTest, ArityAndParamCountMatchSpec) {
    EXPECT_EQ(gate_info(Opcode::X).arity, 1);
    EXPECT_EQ(gate_info(Opcode::X).num_params, 0);
    EXPECT_EQ(gate_info(Opcode::RX).num_params, 1);
    EXPECT_EQ(gate_info(Opcode::U).num_params, 3);
    EXPECT_EQ(gate_info(Opcode::CX).arity, 2);
    EXPECT_EQ(gate_info(Opcode::CCX).arity, 3);
}

TEST(GateTableTest, GenerateMatrixThrowsInternalErrorOnParamCountMismatch) {
    EXPECT_THROW((void)generate_matrix(Opcode::RX, std::vector<double>{}), common::InternalError);
    EXPECT_THROW((void)generate_matrix(Opcode::X, std::vector<double>{1.0}), common::InternalError);
}

// RZZ(t) = CX RZ(t) CX (target = operand 1), the decomposition Qiskit's
// RZZGate and PennyLane's IsingZZ both document: diagonal, symmetric in its
// operands, and e^{-it/2} on the even-parity states |00>, |11>.
TEST(GateTableTest, RzzIsCxRzCxAndDiagonal) {
    using C = std::complex<double>;
    const double theta = 0.83;
    const std::vector<C> rzz = generate_matrix(Opcode::RZZ, std::vector<double>{theta});
    ASSERT_EQ(rzz.size(), 16u);
    const C same = std::exp(C{0.0, -theta / 2});
    const C diff = std::exp(C{0.0, theta / 2});
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            const C expected = r != c ? C{0.0, 0.0} : (r == 0 || r == 3) ? same : diff;
            EXPECT_NEAR(std::abs(rzz[r * 4 + c] - expected), 0.0, 1e-15) << r << "," << c;
        }
    }
    // CX RZ(t) CX in the operand-order bit convention: CX flips bit 1 when
    // bit 0 is set; RZ(t) on bit 1 is diag(e^{-it/2}, e^{it/2}) by that bit.
    const std::vector<C> rz = generate_matrix(Opcode::RZ, std::vector<double>{theta});
    for (std::size_t i = 0; i < 4; ++i) {
        const std::size_t flipped = ((i & 1u) != 0) ? (i ^ 2u) : i;  // CX
        const C phase = rz[((flipped >> 1) & 1u) * 3];               // RZ on bit 1
        EXPECT_NEAR(std::abs(rzz[i * 4 + i] - phase), 0.0, 1e-15) << i;
    }
    EXPECT_FALSE(gate_info(Opcode::RZZ).is_clifford);
    EXPECT_EQ(gate_info(Opcode::RZZ).arity, 2);
    EXPECT_EQ(gate_info(Opcode::RZZ).num_params, 1);
}

TEST(GateTableTest, RzIsNotP) {
    // CLAUDE.md: "RZ is not P. RZ(t) and P(t) differ by a global phase."
    const std::vector<double> theta{1.234};
    const std::vector<std::complex<double>> rz = generate_matrix(Opcode::RZ, theta);
    const std::vector<std::complex<double>> p = generate_matrix(Opcode::P, theta);
    EXPECT_GT(std::abs(rz[0] - p[0]), 1e-6);
}

// docs/BUILD_PLAN.md Phase 7: the single lookup the OpenQASM 2 parser
// uses to map a parsed gate name to an Opcode, reusing this table
// rather than defining a second one.
TEST(GateTableTest, GateInfoByNameFindsEveryOneOfTheNineteenNamedGatesPlusCcx) {
    const struct {
        std::string_view name;
        Opcode opcode;
    } kCases[] = {
        {"x", Opcode::X},     {"y", Opcode::Y},       {"z", Opcode::Z},   {"h", Opcode::H},
        {"s", Opcode::S},     {"sdg", Opcode::Sdg},   {"t", Opcode::T},   {"tdg", Opcode::Tdg},
        {"sx", Opcode::SX},   {"sxdg", Opcode::SXdg}, {"rx", Opcode::RX}, {"ry", Opcode::RY},
        {"rz", Opcode::RZ},   {"p", Opcode::P},       {"u", Opcode::U},   {"cx", Opcode::CX},
        {"cz", Opcode::CZ},   {"swap", Opcode::SWAP}, {"cp", Opcode::CP}, {"ccx", Opcode::CCX},
        {"rzz", Opcode::RZZ},
    };
    for (const auto& c : kCases) {
        const GateInfo* info = gate_info_by_name(c.name);
        ASSERT_NE(info, nullptr) << "name=" << c.name;
        EXPECT_EQ(info->opcode, c.opcode) << "name=" << c.name;
    }
}

TEST(GateTableTest, GateInfoByNameReturnsNullForAnUnknownName) {
    // Real qelib1.inc names with no MENOS opcode (u1/u2/u3, cy, ch, crz,
    // cu1, cu3, cswap), plus barrier/id (handled separately by the
    // parser, not through this table) and garbage input.
    for (std::string_view name : {"u1",
                                  "u2",
                                  "u3",
                                  "cy",
                                  "ch",
                                  "crz",
                                  "cu1",
                                  "cu3",
                                  "cswap",
                                  "barrier",
                                  "id",
                                  "not_a_gate",
                                  ""}) {
        EXPECT_EQ(gate_info_by_name(name), nullptr) << "name=" << name;
    }
}

TEST(GateTableTest, CxOperandOrderControlThenTarget) {
    // operands (control, target): control = bit 0, target = bit 1.
    // Local index 1 = control set, target clear -> control flips target ->
    // maps to local index 3.
    const std::vector<std::complex<double>> cx = generate_matrix(Opcode::CX, {});
    EXPECT_EQ(cx[static_cast<std::size_t>(3 * 4 + 1)], std::complex<double>(1.0, 0.0));
    EXPECT_EQ(cx[static_cast<std::size_t>(1 * 4 + 1)], std::complex<double>(0.0, 0.0));
}

}  // namespace
}  // namespace menos::ir
