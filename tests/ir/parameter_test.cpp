#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <map>
#include <string>

#include "common/error.h"
#include "menos/builder.h"

namespace menos {
namespace {

std::uint64_t bits(double x) {
    return std::bit_cast<std::uint64_t>(x);
}

TEST(ParameterTest, AcceptsIdentifiersOnly) {
    EXPECT_EQ(Parameter("theta").name(), "theta");
    EXPECT_EQ(Parameter("_a1").name(), "_a1");
    EXPECT_THROW(Parameter(""), common::InvalidCircuit);
    EXPECT_THROW(Parameter("1a"), common::InvalidCircuit);
    EXPECT_THROW(Parameter("a-b"), common::InvalidCircuit);
    EXPECT_THROW(Parameter("θ"), common::InvalidCircuit);
}

TEST(ParameterExpressionTest, BareParameterEvaluatesToItsValueBitwise) {
    const ParameterExpression e = Parameter("t");
    for (const double v : {0.0, -0.0, 1.0 / 3.0, -2.5e-300, 1e300}) {
        EXPECT_EQ(bits(e.evaluate({{"t", v}})), bits(v));
        EXPECT_EQ(bits((-e).evaluate({{"t", v}})), bits(-v));
    }
    EXPECT_FALSE(e.constant().has_value());
    EXPECT_FALSE(e.is_constant());
}

TEST(ParameterExpressionTest, ConstantEvaluatesToItself) {
    const ParameterExpression e = -0.0;
    EXPECT_TRUE(e.is_constant());
    EXPECT_EQ(bits(e.evaluate()), bits(-0.0));
}

TEST(ParameterExpressionTest, AffineEvaluationOrderIsDocumented) {
    const Parameter a("a");
    const Parameter b("b");
    const ParameterExpression e = 0.5 + 2.0 * a + b * 3.0 - 0.25;
    ASSERT_EQ(e.terms().size(), 2U);
    EXPECT_EQ(e.terms()[0].name, "a");
    EXPECT_EQ(e.terms()[0].coefficient, 2.0);
    EXPECT_EQ(e.terms()[1].name, "b");
    EXPECT_EQ(e.terms()[1].coefficient, 3.0);
    ASSERT_TRUE(e.constant().has_value());
    EXPECT_EQ(*e.constant(), 0.5 + -0.25);
    const double va = 0.1;
    const double vb = 0.7;
    const double expected = (2.0 * va + 3.0 * vb) + (0.5 + -0.25);
    EXPECT_EQ(bits(e.evaluate({{"a", va}, {"b", vb}})), bits(expected));
}

TEST(ParameterExpressionTest, RepeatedParameterMergesCoefficients) {
    const Parameter a("a");
    const ParameterExpression e = a + a * 2.0 - a;
    ASSERT_EQ(e.terms().size(), 1U);
    EXPECT_EQ(e.terms()[0].coefficient, 1.0 + 2.0 + -1.0);
    // a - a keeps the parameter (coefficient 0): it is still a parameter of
    // the circuit and still needs a binding.
    const ParameterExpression zero = a - a;
    ASSERT_EQ(zero.terms().size(), 1U);
    EXPECT_EQ(zero.terms()[0].coefficient, 0.0);
}

TEST(ParameterExpressionTest, ProductOfTwoParametersIsRefused) {
    const Parameter a("a");
    const Parameter b("b");
    EXPECT_THROW((void)(a * b), common::InvalidCircuit);
    EXPECT_THROW((void)((a + 1.0) * (b + 1.0)), common::InvalidCircuit);
    EXPECT_NO_THROW((void)((a + 1.0) * (2.0 + 3.0)));
}

TEST(ParameterExpressionTest, ScalingScalesConstantToo) {
    const Parameter a("a");
    const ParameterExpression e = (a + 1.0) * 3.0;
    EXPECT_EQ(e.terms()[0].coefficient, 3.0);
    EXPECT_EQ(*e.constant(), 3.0);
}

TEST(ParameterExpressionTest, MissingValueIsAnInternalError) {
    const ParameterExpression e = Parameter("a");
    EXPECT_THROW((void)e.evaluate({}), common::InternalError);
    EXPECT_THROW((void)e.evaluate({{"b", 1.0}}), common::InternalError);
}

}  // namespace
}  // namespace menos
