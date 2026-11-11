#include <cstddef>
#include <string>
#include <utility>

#include "common/error.h"
#include "ir/identifier.h"
#include "menos/builder.h"

namespace menos {

Parameter::Parameter(std::string name) : name_(std::move(name)) {
    if (!ir::is_identifier(name_)) {
        throw common::InvalidCircuit("Parameter: name '" + name_ +
                                     "' is not an identifier ([A-Za-z_][A-Za-z0-9_]*)");
    }
}

ParameterExpression::ParameterExpression(double constant) : constant_(constant) {}

ParameterExpression::ParameterExpression(const Parameter& parameter)
    : terms_{Term{parameter.name(), 1.0}} {}

double ParameterExpression::evaluate(
    const std::map<std::string, double, std::less<>>& values) const {
    double value = 0.0;
    bool started = false;
    for (const Term& term : terms_) {
        const auto it = values.find(term.name);
        common::ensure(it != values.end(),
                       "ParameterExpression::evaluate: parameter '" + term.name + "' has no value");
        const double product = term.coefficient * it->second;
        value = started ? value + product : product;
        started = true;
    }
    if (constant_.has_value()) {
        value = started ? value + *constant_ : *constant_;
    }
    return value;
}

ParameterExpression operator-(const ParameterExpression& a) {
    ParameterExpression out = a;
    for (ParameterExpression::Term& term : out.terms_) {
        term.coefficient = -term.coefficient;
    }
    if (out.constant_.has_value()) {
        out.constant_ = -*out.constant_;
    }
    return out;
}

ParameterExpression operator+(const ParameterExpression& a, const ParameterExpression& b) {
    ParameterExpression out = a;
    for (const ParameterExpression::Term& term : b.terms_) {
        bool merged = false;
        for (ParameterExpression::Term& existing : out.terms_) {
            if (existing.name == term.name) {
                existing.coefficient = existing.coefficient + term.coefficient;
                merged = true;
                break;
            }
        }
        if (!merged) {
            out.terms_.push_back(term);
        }
    }
    if (b.constant_.has_value()) {
        out.constant_ = out.constant_.has_value() ? *out.constant_ + *b.constant_ : *b.constant_;
    }
    return out;
}

ParameterExpression operator-(const ParameterExpression& a, const ParameterExpression& b) {
    return a + (-b);
}

ParameterExpression operator*(const ParameterExpression& a, const ParameterExpression& b) {
    if (!a.is_constant() && !b.is_constant()) {
        throw common::InvalidCircuit(
            "ParameterExpression: a product of two parameter expressions is not affine "
            "(only a constant times an expression is supported)");
    }
    const ParameterExpression& scalar = a.is_constant() ? a : b;
    const ParameterExpression& other = a.is_constant() ? b : a;
    // A constant expression always has a constant (it was built from a
    // double, or from sums and products of constants).
    common::ensure(scalar.constant_.has_value(),
                   "ParameterExpression: a constant expression has no constant");
    const double k = *scalar.constant_;
    ParameterExpression out = other;
    for (ParameterExpression::Term& term : out.terms_) {
        term.coefficient = k * term.coefficient;
    }
    if (out.constant_.has_value()) {
        out.constant_ = k * *out.constant_;
    }
    return out;
}

}  // namespace menos
